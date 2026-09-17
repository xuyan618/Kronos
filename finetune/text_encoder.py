"""
text_encoder.py —— 把文本变成可拼进 predictor 的嵌入向量。

提供两种实现：

1. FinBertTextEncoder（真实路径，用户选定）
   使用中文金融 FinBERT（默认 yiyanghkust/finbert-tone-chinese，bert-base-chinese）取 [CLS] 向量
   （768 维），带中文金融情感/情绪语义，对 A 股因子文本比英文 ProsusAI/finbert 更贴合。
   需要 `pip install transformers`（已加入 requirements.txt）。

2. HashTextEncoder（离线兜底）
   确定性哈希嵌入，**不需要下载任何模型、不需要联网**。
   语义上无意义，仅用于在无 transformers / 离线环境下
   跑通「文本嵌入 -> 融合模型 -> 反向传播」整条通路（脚手架冒烟验证）。

`build_encoder(mode)` 中 mode 取值：
    "finbert" 强制 FinBERT（缺依赖则报错）
    "hash"    强制哈希兜底
    "auto"    优先 FinBERT，不可用时自动降级为 hash（推荐）
"""

import hashlib

import numpy as np

FINBERT_DEFAULT = "yiyanghkust/finbert-tone-chinese"


class HashTextEncoder:
    """确定性哈希嵌入：离线兜底，用于验证通路而非提供语义。"""

    def __init__(self, dim=128):
        self.dim = dim

    @property
    def dim_out(self):
        return self.dim

    def encode(self, texts, batch_size=256):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            vec = np.zeros(self.dim, dtype=np.float32)
            toks = str(t).split()
            if not toks:
                toks = ["<empty>"]
            for tok in toks:
                h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
                idx = h % self.dim
                sign = 1.0 if ((h >> 8) & 1) else -1.0
                vec[idx] += sign
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            out[i] = vec
        return out


class FinBertTextEncoder:
    """FinBERT [CLS] 句向量（768 维）。"""

    def __init__(self, model_name=FINBERT_DEFAULT, device=None, max_length=64,
                 batch_size=16, pooling="cls"):
        try:
            from transformers import AutoTokenizer, AutoModel
        except ImportError as e:
            raise ImportError(
                "FinBertTextEncoder 需要 transformers，请先 `pip install transformers`；"
                "或改用 HashTextEncoder / mode='auto' 自动降级。"
            ) from e

        import torch

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self.pooling = pooling
        self.torch = torch

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.dim = self.model.config.hidden_size

    @property
    def dim_out(self):
        return self.dim

    def encode(self, texts, batch_size=None):
        import torch

        bs = batch_size or self.batch_size
        embs = []
        for start in range(0, len(texts), bs):
            chunk = [str(t) if t else "" for t in texts[start:start + bs]]
            enc = self.tokenizer(
                chunk, padding=True, truncation=True,
                max_length=self.max_length, return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                out = self.model(**enc).last_hidden_state  # [B, L, H]
            if self.pooling == "cls":
                pooled = out[:, 0, :]
            else:  # mean pooling over non-pad tokens
                mask = enc["attention_mask"].unsqueeze(-1).to(out.dtype)
                pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
            embs.append(pooled.detach().cpu().float().numpy())
        return np.concatenate(embs, axis=0) if embs else np.zeros((0, self.dim), dtype=np.float32)


def build_encoder(mode="auto", hash_dim=128, finbert_model=FINBERT_DEFAULT, device=None):
    """
    构建文本编码器。

    mode: "finbert" | "hash" | "auto"
    """
    if mode == "hash":
        print("[text_encoder] mode=hash (离线兜底嵌入)")
        return HashTextEncoder(dim=hash_dim)

    if mode == "finbert":
        enc = FinBertTextEncoder(model_name=finbert_model, device=device)
        print(f"[text_encoder] mode=finbert ({finbert_model}, dim={enc.dim_out}, device={enc.device})")
        return enc

    # auto: 优先 FinBERT，失败降级
    try:
        enc = FinBertTextEncoder(model_name=finbert_model, device=device)
        print(f"[text_encoder] mode=auto -> finbert (dim={enc.dim_out}, device={enc.device})")
        return enc
    except Exception as e:
        print(f"[text_encoder] FinBERT 不可用（{type(e).__name__}: {e}），降级为 hash 兜底嵌入")
        print("[text_encoder] 提示：安装 `pip install transformers` 后即可使用真实 FinBERT 嵌入")
        return HashTextEncoder(dim=hash_dim)
