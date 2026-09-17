"""
KronosFusion —— 前期融合（Early Fusion）模型。

把文本嵌入当作**额外 token** 拼进 predictor 的输入序列，让 Transformer 在
同一条序列内对「价格 token」与「文本 token」做跨模态联合注意力，
从而用文本（新闻 / 因子文本）直接条件化对下一个价格 token 的预测。

设计要点
--------
1. 底层复用预训练 Kronos（backbone），权重原样加载，只新增文本通道，
   因此不破坏原有纯价格预测能力。
2. 新增模块：
   - `text_proj`：Linear(text_dim -> d_model)，把 FinBERT 等文本嵌入投到模型维度；
   - `modality_emb`：可学习的模态标识（0=文本, 1=价格）；
   - `null_text`：未来日期没有文本时使用可学习占位向量。
3. **融合布局（关键）**：Kronos 的自注意力是 `is_causal=True`。
   - `interleave`（默认，因果正确）：`[t_0, p_0, t_1, p_1, ...]`
     price_t 位于 2t+1，只能看到 text_0..text_t，不会泄露未来文本。
   - `prepend`：`[t_0..t_{T-1}, p_0..p_{T-1}]`
     price_t 会看到全部文本（含未来），**存在未来泄漏**，仅用于消融对照。
4. 文本 token 与同日价格 token 共享该日的 TemporalEmbedding（同一天），
   模态差异由 `modality_emb` 区分。
5. 只在**价格位置**计算 logits／loss，目标仍是预测下一个价格 token，
   因此可直接复用 `DualHead.compute_loss`。
"""

import os
import sys
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from model.module import RMSNorm, TransformerBlock  # noqa: E402  (保持与 kronos.py 一致的相对导入环境)
from model.kronos import Kronos  # noqa: E402

try:  # 推理采样复用官方实现，避免两份逻辑漂移
    from model.kronos import sample_from_logits
except Exception:  # pragma: no cover
    sample_from_logits = None


class KronosFusion(nn.Module, PyTorchModelHubMixin):
    """在预训练 Kronos 之上增加文本 token 通道的前期融合模型。"""

    def __init__(
        self,
        s1_bits,
        s2_bits,
        n_layers,
        d_model,
        n_heads,
        ff_dim,
        ffn_dropout_p,
        attn_dropout_p,
        resid_dropout_p,
        token_dropout_p,
        learn_te,
        text_dim=768,
        fusion_mode="interleave",
        use_modality_emb=True,
    ):
        super().__init__()
        assert fusion_mode in ("interleave", "prepend"), \
            f"fusion_mode must be 'interleave' or 'prepend', got {fusion_mode}"

        self.s1_bits = s1_bits
        self.s2_bits = s2_bits
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.ff_dim = ff_dim
        self.text_dim = text_dim
        self.fusion_mode = fusion_mode
        self.use_modality_emb = use_modality_emb
        self.s1_vocab_size = 2 ** s1_bits

        # 预训练 backbone（结构与权重均与 Kronos 一致）
        self.backbone = Kronos(
            s1_bits, s2_bits, n_layers, d_model, n_heads, ff_dim,
            ffn_dropout_p, attn_dropout_p, resid_dropout_p,
            token_dropout_p, learn_te,
        )

        # ---- 新增的文本通道 ----
        self.text_proj = nn.Linear(text_dim, d_model)
        if use_modality_emb:
            self.modality_emb = nn.Parameter(torch.zeros(2, d_model))  # 0=text, 1=price
        # 未来日期（预测区间）没有真实文本，用可学习占位
        self.null_text = nn.Parameter(torch.zeros(text_dim))

        nn.init.xavier_normal_(self.text_proj.weight)
        if self.text_proj.bias is not None:
            nn.init.zeros_(self.text_proj.bias)

        # ---- 收益率回归头：直接预测「次日收益率(带符号)」 ----
        # 解决 close 水平预测器天然偏均值回归、方向符号随体制漂移的问题。
        # 训练时作为辅助监督（与价格 token CE 联合），回测时此头输出即方向信号。
        self.return_head = nn.Linear(d_model, 1)
        nn.init.zeros_(self.return_head.weight)
        nn.init.zeros_(self.return_head.bias)

    # ---------- 便捷代理：让训练脚本能像用 Kronos 一样访问 head / embedding ----------
    @property
    def head(self):
        return self.backbone.head

    @property
    def embedding(self):
        return self.backbone.embedding

    # ---------- 融合核心 ----------
    def _fuse(self, price_emb, text_emb, mode):
        """price_emb / text_emb: [B, T, d] -> fused: [B, 2T, d]"""
        B, T, d = price_emb.shape
        if self.use_modality_emb:
            price_emb = price_emb + self.modality_emb[1]
            text_emb = text_emb + self.modality_emb[0]
        if mode == "prepend":
            return torch.cat([text_emb, price_emb], dim=1)
        # interleave: [t_0, p_0, t_1, p_1, ...]
        stacked = torch.stack([text_emb, price_emb], dim=2)  # [B, T, 2, d]
        return stacked.reshape(B, 2 * T, d)

    def _price_positions(self, fused, T, mode):
        """从融合序列中取出价格位置（用于算 logits / loss）。"""
        if mode == "prepend":
            return fused[:, T:, :]
        return fused[:, 1::2, :]

    def _extend_padding_mask(self, padding_mask, T, mode):
        """文本 token 从不是 padding，需要把 mask 按融合布局展开。"""
        if padding_mask is None:
            return None
        B = padding_mask.shape[0]
        if mode == "prepend":
            text_pad = torch.zeros((B, T), dtype=padding_mask.dtype, device=padding_mask.device)
            return torch.cat([text_pad, padding_mask], dim=1)
        text_pad = torch.zeros_like(padding_mask)
        stacked = torch.stack([text_pad, padding_mask], dim=2)  # [B, T, 2]
        return stacked.reshape(B, 2 * T)

    def _build_fused(self, s1_ids, s2_ids, stamp, text_emb, padding_mask, mode):
        """构造融合后的输入序列（含时间嵌入与模态标识）。"""
        b = self.backbone
        x = b.embedding([s1_ids, s2_ids])  # [B, T, d]
        if stamp is not None:
            te = b.time_emb(stamp)  # 当日时间嵌入
            x = x + te
        x = b.token_drop(x)

        T = x.shape[1]
        if text_emb is None:
            # 无文本时退化为纯价格序列（保证接口兼容）
            return x, padding_mask, T

        t = self.text_proj(text_emb)
        if stamp is not None:
            # 文本 token 与同日价格 token 共享当日时间嵌入
            t = t + te

        fused = self._fuse(x, t, mode)
        fused_mask = self._extend_padding_mask(padding_mask, T, mode)
        return fused, fused_mask, T

    def _run_transformer(self, fused, fused_mask):
        b = self.backbone
        for layer in b.transformer:
            fused = layer(fused, key_padding_mask=fused_mask)
        return b.norm(fused)

    # ---------- 前向 ----------
    def forward(
        self,
        s1_ids,
        s2_ids,
        stamp=None,
        text_emb=None,
        padding_mask=None,
        use_teacher_forcing=False,
        s1_targets=None,
        fusion_mode=None,
    ):
        """
        Args:
            s1_ids / s2_ids: [B, T] 价格 token ID
            stamp: [B, T, 5] 时间特征
            text_emb: [B, T, text_dim] 与价格 token 逐日对齐的文本嵌入
        Returns:
            (s1_logits, s2_logits)：仅在**价格位置**上输出，形状 [B, T, vocab]
        """
        mode = fusion_mode or self.fusion_mode
        b = self.backbone
        price_padding_mask = padding_mask  # 保留原始（价格级）mask 给依赖层

        fused, fused_mask, T = self._build_fused(s1_ids, s2_ids, stamp, text_emb, padding_mask, mode)
        fused = self._run_transformer(fused, fused_mask)

        price_ctx = self._price_positions(fused, T, mode)  # [B, T, d]
        s1_logits = b.head(price_ctx)

        if use_teacher_forcing:
            sibling_embed = b.embedding.emb_s1(s1_targets)
        else:
            s1_probs = F.softmax(s1_logits.detach(), dim=-1)
            sample_s1_ids = torch.multinomial(
                s1_probs.view(-1, self.s1_vocab_size), 1
            ).view(s1_ids.shape)
            sibling_embed = b.embedding.emb_s1(sample_s1_ids)

        x2 = b.dep_layer(price_ctx, sibling_embed, key_padding_mask=price_padding_mask)
        s2_logits = b.head.cond_forward(x2)
        ret_logits = self.return_head(price_ctx)  # [B, T, 1] 次日收益率预测（带符号）
        return s1_logits, s2_logits, ret_logits

    # ---------- 自回归推理 ----------
    def decode_s1(self, s1_ids, s2_ids, stamp=None, text_emb=None, padding_mask=None):
        """返回价格位置上的 s1 logits 与价格上下文。"""
        mode = self.fusion_mode
        fused, fused_mask, T = self._build_fused(s1_ids, s2_ids, stamp, text_emb, padding_mask, mode)
        fused = self._run_transformer(fused, fused_mask)
        price_ctx = self._price_positions(fused, T, mode)
        s1_logits = self.backbone.head(price_ctx)
        return s1_logits, price_ctx

    def decode_s2(self, price_ctx, s1_ids, padding_mask=None):
        sibling_embed = self.backbone.embedding.emb_s1(s1_ids)
        x2 = self.backbone.dep_layer(price_ctx, sibling_embed, key_padding_mask=padding_mask)
        return self.backbone.head.cond_forward(x2)

    def predict_return(self, s1_ids, s2_ids, stamp=None, text_emb=None, padding_mask=None):
        """返回每个价格位置上的「次日收益率」预测（带符号标量）。

        取最后一个价格位置 -> 预测 context 之后那一天的收益率，作回测方向信号。
        """
        s1_logits, price_ctx = self.decode_s1(s1_ids, s2_ids, stamp, text_emb, padding_mask)
        ret = self.return_head(price_ctx)  # [B, T, 1]
        return ret[:, -1, 0]  # [B] 预测 context 后一天的收益率


def build_fusion_from_pretrained(pretrained_path, text_dim=768, fusion_mode="interleave",
                                 use_modality_emb=True, device=None):
    """
    从预训练 Kronos 权重构建 KronosFusion：
    backbone 权重原样加载（保留原预测能力），文本通道随机初始化后随训练学习。
    """
    k = Kronos.from_pretrained(pretrained_path)
    fusion = KronosFusion(
        s1_bits=k.s1_bits,
        s2_bits=k.s2_bits,
        n_layers=k.n_layers,
        d_model=k.d_model,
        n_heads=k.n_heads,
        ff_dim=k.ff_dim,
        ffn_dropout_p=k.ffn_dropout_p,
        attn_dropout_p=k.attn_dropout_p,
        resid_dropout_p=k.resid_dropout_p,
        token_dropout_p=k.token_dropout_p,
        learn_te=k.learn_te,
        text_dim=text_dim,
        fusion_mode=fusion_mode,
        use_modality_emb=use_modality_emb,
    )
    missing, unexpected = fusion.backbone.load_state_dict(k.state_dict(), strict=False)
    if missing:
        print(f"[fusion] backbone missing keys: {len(missing)}")
    if unexpected:
        print(f"[fusion] backbone unexpected keys: {len(unexpected)}")
    if device is not None:
        fusion = fusion.to(device)
    return fusion


def auto_regressive_inference_fused(
    tokenizer,
    model,
    x,
    x_stamp,
    y_stamp,
    text_emb,
    pred_len,
    max_context=512,
    clip=5,
    T=1.0,
    top_k=0,
    top_p=0.99,
    sample_count=1,
    verbose=False,
):
    """
    带文本的前期融合自回归推理。

    text_emb: [B, T_ctx, text_dim]，覆盖历史窗口每一天。
    预测区间（未来）没有真实文本，使用模型内可学习的 `null_text` 占位。
    """
    assert sample_from_logits is not None, "sample_from_logits 不可用"
    from tqdm import trange

    with torch.no_grad():
        x = torch.clip(x, -clip, clip)
        device = x.device

        # 多样本并行（与官方实现一致）
        x = x.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, x.size(1), x.size(2)).to(device)
        x_stamp = x_stamp.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, x_stamp.size(1), x_stamp.size(2)).to(device)
        y_stamp = y_stamp.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, y_stamp.size(1), y_stamp.size(2)).to(device)
        text_emb = text_emb.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, text_emb.size(1), text_emb.size(2)).to(device)

        x_token = tokenizer.encode(x, half=True)  # (pre, post)，各 [B, T]

        initial_seq_len = x.size(1)
        batch_size = x_token[0].size(0)
        text_dim = text_emb.size(2)
        total_seq_len = initial_seq_len + pred_len
        full_stamp = torch.cat([x_stamp, y_stamp], dim=1)

        generated_pre = x_token[0].new_empty(batch_size, pred_len)
        generated_post = x_token[1].new_empty(batch_size, pred_len)

        # 价格 token 缓冲
        pre_buffer = x_token[0].new_zeros(batch_size, max_context)
        post_buffer = x_token[1].new_zeros(batch_size, max_context)
        # 文本缓冲（与价格缓冲按天对齐）；未来日期用 null_text
        text_buffer = text_emb.new_zeros(batch_size, max_context, text_dim)

        buffer_len = min(initial_seq_len, max_context)
        if buffer_len > 0:
            start_idx = max(0, initial_seq_len - max_context)
            pre_buffer[:, :buffer_len] = x_token[0][:, start_idx:start_idx + buffer_len]
            post_buffer[:, :buffer_len] = x_token[1][:, start_idx:start_idx + buffer_len]
            text_buffer[:, :buffer_len] = text_emb[:, start_idx:start_idx + buffer_len]

        ran = trange if verbose else range
        for i in ran(pred_len):
            current_seq_len = initial_seq_len + i
            window_len = min(current_seq_len, max_context)

            if current_seq_len <= max_context:
                pre_in = pre_buffer[:, :window_len]
                post_in = post_buffer[:, :window_len]
                txt_in = text_buffer[:, :window_len]
            else:
                pre_in, post_in, txt_in = pre_buffer, post_buffer, text_buffer

            context_end = current_seq_len
            context_start = max(0, context_end - window_len)
            current_stamp = full_stamp[:, context_start:context_end, :].contiguous()

            s1_logits, price_ctx = model.decode_s1(pre_in, post_in, current_stamp, txt_in)
            s1_logits = s1_logits[:, -1, :]  # 最后一个价格位置 -> 预测下一天
            sample_pre = sample_from_logits(s1_logits, temperature=T, top_k=top_k, top_p=top_p, sample_logits=True)

            s2_logits = model.decode_s2(price_ctx, sample_pre)
            s2_logits = s2_logits[:, -1, :]
            sample_post = sample_from_logits(s2_logits, temperature=T, top_k=top_k, top_p=top_p, sample_logits=True)

            generated_pre[:, i] = sample_pre.squeeze(-1)
            generated_post[:, i] = sample_post.squeeze(-1)

            # 新生成的一天：文本未知，填 null_text
            if current_seq_len < max_context:
                pre_buffer[:, current_seq_len] = sample_pre.squeeze(-1)
                post_buffer[:, current_seq_len] = sample_post.squeeze(-1)
                text_buffer[:, current_seq_len] = model.null_text
            else:
                pre_buffer.copy_(torch.roll(pre_buffer, shifts=-1, dims=1))
                post_buffer.copy_(torch.roll(post_buffer, shifts=-1, dims=1))
                text_buffer.copy_(torch.roll(text_buffer, shifts=-1, dims=1))
                pre_buffer[:, -1] = sample_pre.squeeze(-1)
                post_buffer[:, -1] = sample_post.squeeze(-1)
                text_buffer[:, -1] = model.null_text

        full_pre = torch.cat([x_token[0], generated_pre], dim=1)
        full_post = torch.cat([x_token[1], generated_post], dim=1)

        context_start = max(0, total_seq_len - max_context)
        input_tokens = [
            full_pre[:, context_start:total_seq_len].contiguous(),
            full_post[:, context_start:total_seq_len].contiguous(),
        ]
        z = tokenizer.decode(input_tokens, half=True)
        z = z.reshape(-1, sample_count, z.size(1), z.size(2))
        preds = z.cpu().numpy()
        preds = np.mean(preds, axis=1)
        return preds
