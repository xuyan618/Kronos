"""
KronosTokenizerExtended: 在预训练 KronosTokenizer 权重基础上引入新维度。

设计要点（对应需求「在原有权重上引入新维度，改变权重」）:
- BSQ 码本维度 (s1_bits + s2_bits) 保持不变 → token 词表空间不变
  → Kronos predictor 的 HierarchicalEmbedding 结构无需改动，权重可直接复用。
- 仅扩展 tokenizer 的「输入嵌入」与「输出头」为双分支:
    * embed_orig / head_orig: 加载预训练原权重（OHLCV+amount 的 6 维）
    * embed_new  / head_new : 随机初始化（新维度，例如 outstanding_share）
- 中间的 encoder / BSQ / decoder 全部复用预训练权重。
- 前向: z = embed_orig(x_orig) + embed_new(x_new)，其余与原始一致。
- 因此「新维度」被融合进同一套 token 语义中，predictor 只需在微调时适应新语义。
"""

import torch
import torch.nn as nn

from model.kronos import KronosTokenizer
from model.module import TransformerBlock, BSQuantizer


class KronosTokenizerExtended(KronosTokenizer):
    def __init__(self, d_in, d_model, n_heads, ff_dim, n_enc_layers, n_dec_layers,
                 ffn_dropout_p, attn_dropout_p, resid_dropout_p, s1_bits, s2_bits,
                 beta, gamma0, gamma, zeta, group_size, d_in_orig=6, d_in_new=1):
        # 不调用 super().__init__（它会创建我们用不到的单分支 embed/head），
        # 只初始化 nn.Module 并手动重建双分支结构。
        nn.Module.__init__(self)

        self.d_in = d_in
        self.d_model = d_model
        self.n_heads = n_heads
        self.ff_dim = ff_dim
        self.enc_layers = n_enc_layers
        self.dec_layers = n_dec_layers
        self.ffn_dropout_p = ffn_dropout_p
        self.attn_dropout_p = attn_dropout_p
        self.resid_dropout_p = resid_dropout_p
        self.s1_bits = s1_bits
        self.s2_bits = s2_bits
        self.codebook_dim = s1_bits + s2_bits
        self.d_in_orig = d_in_orig
        self.d_in_new = d_in_new

        # ---- 双分支输入嵌入 ----
        self.embed_orig = nn.Linear(self.d_in_orig, self.d_model)   # 预训练权重
        self.embed_new = nn.Linear(self.d_in_new, self.d_model)    # 随机初始化

        # ---- 复用预训练的 encoder / quantizer / decoder ----
        self.encoder = nn.ModuleList([
            TransformerBlock(self.d_model, self.n_heads, self.ff_dim, self.ffn_dropout_p,
                             self.attn_dropout_p, self.resid_dropout_p)
            for _ in range(self.enc_layers - 1)
        ])
        self.decoder = nn.ModuleList([
            TransformerBlock(self.d_model, self.n_heads, self.ff_dim, self.ffn_dropout_p,
                             self.attn_dropout_p, self.resid_dropout_p)
            for _ in range(self.dec_layers - 1)
        ])
        self.quant_embed = nn.Linear(in_features=self.d_model, out_features=self.codebook_dim)
        self.post_quant_embed_pre = nn.Linear(in_features=self.s1_bits, out_features=self.d_model)
        self.post_quant_embed = nn.Linear(in_features=self.codebook_dim, out_features=self.d_model)
        self.tokenizer = BSQuantizer(self.s1_bits, self.s2_bits, beta, gamma0, gamma, zeta, group_size)

        # ---- 双分支输出头 ----
        self.head_orig = nn.Linear(self.d_model, self.d_in_orig)   # 预训练权重
        self.head_new = nn.Linear(self.d_model, self.d_in_new)     # 随机初始化

    # ------------------------------------------------------------------
    def _split(self, x):
        return x[..., :self.d_in_orig], x[..., self.d_in_orig:]

    def forward(self, x):
        x_orig, x_new = self._split(x)
        z = self.embed_orig(x_orig) + self.embed_new(x_new)

        for layer in self.encoder:
            z = layer(z)

        z = self.quant_embed(z)
        bsq_loss, quantized, z_indices = self.tokenizer(z)

        quantized_pre = quantized[:, :, :self.s1_bits]
        z_pre = self.post_quant_embed_pre(quantized_pre)
        z = self.post_quant_embed(quantized)

        for layer in self.decoder:
            z_pre = layer(z_pre)
        for layer in self.decoder:
            z = layer(z)

        z_pre = torch.cat([self.head_orig(z_pre), self.head_new(z_pre)], dim=-1)
        z = torch.cat([self.head_orig(z), self.head_new(z)], dim=-1)
        return (z_pre, z), bsq_loss, quantized, z_indices

    def encode(self, x, half=False):
        x_orig, x_new = self._split(x)
        z = self.embed_orig(x_orig) + self.embed_new(x_new)
        for layer in self.encoder:
            z = layer(z)
        z = self.quant_embed(z)
        bsq_loss, quantized, z_indices = self.tokenizer(z, half=half, collect_metrics=False)
        return z_indices

    def decode(self, x, half=False):
        quantized = self.indices_to_bits(x, half)
        z = self.post_quant_embed(quantized)
        for layer in self.decoder:
            z = layer(z)
        z = torch.cat([self.head_orig(z), self.head_new(z)], dim=-1)
        return z

    # ------------------------------------------------------------------
    @classmethod
    def from_pretrained_extended(cls, pretrained_path, d_in_new, d_in_orig=6):
        """
        加载预训练 base tokenizer，复用其全部权重（除两条新分支外），
        新分支保持随机初始化。
        """
        base = KronosTokenizer.from_pretrained(pretrained_path)

        bsq = base.tokenizer.bsq  # BinarySphericalQuantizer
        inst = cls(
            d_in=d_in_orig + d_in_new,
            d_model=base.d_model,
            n_heads=base.n_heads,
            ff_dim=base.ff_dim,
            n_enc_layers=base.enc_layers,
            n_dec_layers=base.dec_layers,
            ffn_dropout_p=base.ffn_dropout_p,
            attn_dropout_p=base.attn_dropout_p,
            resid_dropout_p=base.resid_dropout_p,
            s1_bits=base.s1_bits,
            s2_bits=base.s2_bits,
            beta=bsq.beta,
            gamma0=bsq.gamma0,
            gamma=bsq.gamma,
            zeta=bsq.zeta,
            group_size=bsq.group_size,
            d_in_orig=d_in_orig,
            d_in_new=d_in_new,
        )

        # 复制全部预训练权重（除 embed_new / head_new）
        with torch.no_grad():
            inst.embed_orig.weight.copy_(base.embed.weight)
            inst.embed_orig.bias.copy_(base.embed.bias)
            inst.head_orig.weight.copy_(base.head.weight)
            inst.head_orig.bias.copy_(base.head.bias)
            inst.encoder.load_state_dict(base.encoder.state_dict())
            inst.decoder.load_state_dict(base.decoder.state_dict())
            inst.quant_embed.load_state_dict(base.quant_embed.state_dict())
            inst.post_quant_embed_pre.load_state_dict(base.post_quant_embed_pre.state_dict())
            inst.post_quant_embed.load_state_dict(base.post_quant_embed.state_dict())
            inst.tokenizer.load_state_dict(base.tokenizer.state_dict())
        # embed_new / head_new 保持随机初始化
        inst.eval()
        return inst
