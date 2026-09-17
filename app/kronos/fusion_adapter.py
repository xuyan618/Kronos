"""生产适配器：把 `kronos_direction` 改为由训练好的 KronosFusion（价格+文本）模型
的收益率回归头 ``predict_return`` 产出，与回测 ``finetune/backtest_predictor_fusion.py``
的推理严格一致。

设计要点
--------
- 仅当环境变量 ``USE_FUSION_MODEL=1`` 时由 ``app/pipeline.py`` 懒加载调用，
  不设置时生产链路保持原样（原始 KronosForecastAdapter + FinBERT 情感）。
- 推理时**实时**为上下文窗口的每一日构建文本嵌入：从 ``sequoia_v2.db`` 取该标的
  的因子/新闻文本，用 FinBERT（768 维 [CLS]）编码，按日期对齐。文本缺失日填零向量。
  对齐方式复刻回测：精确按日匹配（与回测一致；训练期用的 -3..0 容差不影响这里）。
- 价格归一化（lookback 均值/标准差、clip 5.0）+ tokenizer 编码 完全照搬回测。
- 输出 ``predict_return`` ≈ 次日收益率×RETURN_TARGET_SCALE(默认 100)。
  单标的决策需把它标定到 [-1,1]：``direction = clip(pred_frac / FUSION_DIRECTION_SCALE, -1, 1)``，
  其中 ``pred_frac = pred / 100``，``FUSION_DIRECTION_SCALE`` 默认 0.0134（校准结论：≈2σ，预测 +1.34% → 满仓多）。
  ⚠️ 该刻度是单标的下的标定旋钮，需按实盘校准；回测本身是截面排序、不依赖此刻度。

依赖：torch / transformers（FinBERT），以及 finetune 下的 model / text_source / text_encoder。
这些 import 全部延迟到首次调用，故本模块被 import 不会强制拉起重依赖。
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# 配置（均可用环境变量覆盖）
# ----------------------------------------------------------------------------
FUSION_L = 90                      # 上下文长度（与训练/回测一致）
FUSION_CLIP = 5.0                 # z-score 截断（与回测一致）
RETURN_TARGET_SCALE = float(os.environ.get("RETURN_TARGET_SCALE", "100"))  # 须与训练一致
FUSION_DIRECTION_SCALE = float(os.environ.get("FUSION_DIRECTION_SCALE", "0.0134"))  # 单标的标定刻度（校准结论：≈2σ）

_TEXT_ENCODER_MODE = "finbert"     # 强制 FinBERT(768)，避免 hash(128) 维度不匹配 text_proj
_FINBERT_MODEL = os.environ.get("TEXT_FINBERT_MODEL", "yiyanghkust/finbert-tone-chinese")

# 单例缓存
_model = None
_tokenizer = None
_cfg = None
_text_encoder = None
_text_lookup: Dict[str, Dict[str, np.ndarray]] = {}


def _repo_root() -> str:
    # app/kronos/fusion_adapter.py 位于 仓库根/app/kronos，
    # 从 app/kronos 上溯两级（app/kronos -> app -> 仓库根）即到仓库根。
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(here))


def _ensure_finetune_importable() -> None:
    root = _repo_root()
    finetune = os.path.join(root, "finetune")
    for p in (root, finetune):
        if p not in sys.path:
            sys.path.insert(0, p)


def _model_dir() -> str:
    return os.environ.get(
        "SEQUOIA_FUSION_MODEL_DIR",
        os.path.join(_repo_root(), "outputs", "models",
                     "sequoia_predictor_fusion", "checkpoints", "best_model"),
    )


def _db_path() -> str:
    # 默认指向共享数据库 BigAData/sequoia_v2.db（与 sequoia_db_loader 一致）；
    # 可用环境变量 SEQUOIA_V2_DB 覆盖。
    return os.environ.get(
        "SEQUOIA_V2_DB",
        "/Users/xuyan/Desktop/BigAData/sequoia_v2.db",
    )


def _load_model_and_tokenizer() -> None:
    """加载 KronosFusion + 扩展 tokenizer（进程内仅一次）。"""
    global _model, _tokenizer, _cfg
    if _model is not None:
        return

    _ensure_finetune_importable()
    import torch
    from config_sequoia_fusion import get_config
    from model.kronos_fusion import KronosFusion, build_fusion_from_pretrained
    from model_factory import build_tokenizer
    from safetensors.torch import load_file

    # 扩展 tokenizer 需要 SEQUOIA=1 才走扩展维度路径（与训练/回测一致）
    os.environ.setdefault("SEQUOIA", "1")
    # FinBERT 走镜像，避免 huggingface.co 拉取失败
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    cfg = get_config()
    _cfg = cfg
    device = "cuda" if torch.cuda.is_available() else "cpu"

    best = _model_dir()
    if not os.path.exists(best):
        raise FileNotFoundError(f"融合模型权重不存在: {best}")

    # 优先 from_pretrained；失败则回退到「base Kronos 构建 + 加载 safetensors」
    model = None
    try:
        model = KronosFusion.from_pretrained(best)
    except Exception as e:  # noqa: BLE001
        print(f"[fusion-adapter] from_pretrained 失败({e})，回退构建")
        model = build_fusion_from_pretrained(
            cfg.pretrained_predictor_path,
            text_dim=int(getattr(cfg, "text_dim", 768)),
            fusion_mode=getattr(cfg, "fusion_mode", "interleave"),
            use_modality_emb=bool(getattr(cfg, "use_modality_emb", True)),
            device=device,
        )
        sd = load_file(os.path.join(best, "model.safetensors"))
        model.load_state_dict(sd, strict=False)

    _model = model.to(device).eval()
    # build_tokenizer 把 config 当 dict 用（config.get / config['...']），
    # 与回测一致地传 cfg.__dict__（而非 Config 对象）。
    _tokenizer = build_tokenizer(cfg.__dict__, finetuned=True).to(device).eval()
    _model._device = device  # type: ignore[attr-defined]


def _get_text_lookup(symbol: str) -> Dict[str, np.ndarray]:
    """构建并缓存 {date_str: emb}（FinBERT 768 维），对齐回测的精确日匹配。"""
    global _text_encoder, _text_lookup
    if symbol in _text_lookup:
        return _text_lookup[symbol]

    _ensure_finetune_importable()
    from text_encoder import build_encoder
    from text_source import build_text_by_symbol

    if _text_encoder is None:
        # 强制 finbert：训练用的是 768 维 FinBERT；用 auto 可能因缺依赖降级到 128 维 hash
        # 导致与模型 text_proj(768->d_model) 维度不匹配。
        _text_encoder = build_encoder(mode=_TEXT_ENCODER_MODE, finbert_model=_FINBERT_MODEL)

    tb = build_text_by_symbol(_db_path(), symbols=[symbol], sources=("factor", "news"))
    dmap = tb.get(symbol, {})
    lut: Dict[str, np.ndarray] = {}
    if dmap:
        uniq = sorted(set(dmap.values()))
        embs = _text_encoder.encode(uniq)
        emb_map = {t: e for t, e in zip(uniq, embs)}
        lut = {d: emb_map[t] for d, t in dmap.items()}
    _text_lookup[symbol] = lut
    return lut


def predict_fusion_direction(symbol: str, history: pd.DataFrame, as_of_date: str, scale: float | None = None) -> float:
    """返回该标的在 ``as_of_date`` 之后一天的融合模型方向信号 ∈ [-1, 1]。

    - 历史不足 FUSION_L+1 行时返回 0.0（hold）。
    - 复刻 ``backtest_predictor_fusion.py`` 的逐标推理（归一化、tokenizer、predict_return）。
    """
    _load_model_and_tokenizer()
    import torch

    device = _model._device  # type: ignore[attr-defined]
    cfg = _cfg
    feature_list = list(cfg.feature_list)
    time_feature_list = list(cfg.time_feature_list)

    df = history.copy()
    if "timestamp" not in df.columns:
        raise ValueError("history 必须包含 'timestamp' 列")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
    df = df.sort_values("timestamp").reset_index(drop=True)

    # 派生模型所需价格列（生产 Yahoo 数据可能缺 amt / outstanding_share）
    if "volume" in df.columns and "vol" not in df.columns:
        df["vol"] = df["volume"]
    if "amt" not in df.columns:
        df["amt"] = df["vol"] * df["close"]
    if "outstanding_share" not in df.columns:
        # 近似：归一后恒为 0（训练用真实流通股本；生产无此字段时的保守近似）
        df["outstanding_share"] = 0.0

    ts = df["timestamp"]
    df["minute"] = ts.dt.minute
    df["hour"] = ts.dt.hour
    df["weekday"] = ts.dt.weekday
    df["day"] = ts.dt.day
    df["month"] = ts.dt.month
    df = df.set_index("timestamp")
    df.index.name = "datetime"

    ad = pd.Timestamp(as_of_date)
    if ad not in df.index:
        return 0.0
    idx = df.index.get_loc(ad)
    if idx < FUSION_L:
        return 0.0  # 上下文不足，hold

    win = df.iloc[idx - FUSION_L: idx + 1]
    feat = win[feature_list].values.astype(np.float64)
    past = feat[:FUSION_L]
    mu = past.mean(0)
    sd = past.std(0)
    x = (feat - mu) / (sd + 1e-5)
    x = np.clip(x, -FUSION_CLIP, FUSION_CLIP)
    x_ctx = x[:FUSION_L]
    stamp = win[time_feature_list].values.astype(np.float64)[:FUSION_L]

    dates = [d.strftime("%Y-%m-%d") for d in win.index]
    tlookup = _get_text_lookup(symbol)
    txt = np.zeros((FUSION_L, _text_encoder.dim_out), dtype=np.float32)
    for i in range(FUSION_L):
        e = tlookup.get(dates[i])
        if e is not None:
            txt[i] = e

    xs = torch.tensor(np.stack([x_ctx]), dtype=torch.float32, device=device)
    ss = torch.tensor(np.stack([stamp]), dtype=torch.float32, device=device)
    ts_t = torch.tensor(np.stack([txt]), dtype=torch.float32, device=device)
    t0, t1 = _tokenizer.encode(xs, half=True)
    with torch.no_grad():
        ret = _model.predict_return(t0, t1, ss, text_emb=ts_t)
    pred_ret_pct = float(ret.detach().cpu().numpy()[0])      # ≈ 次日收益率 × 100
    pred_ret_frac = pred_ret_pct / RETURN_TARGET_SCALE        # ≈ 次日收益率

    # 单标的标定到 [-1, 1]（截面排序以外的使用场景需要）
    scale = scale if scale is not None else FUSION_DIRECTION_SCALE
    direction = max(-1.0, min(1.0, pred_ret_frac / scale)) if scale > 0 else 0.0
    return float(direction)


def last_predicted_return(symbol: str, history: pd.DataFrame, as_of_date: str, scale: float | None = None):
    """调试用：返回 (direction, predicted_return_frac)，便于校准 FUSION_DIRECTION_SCALE。"""
    _load_model_and_tokenizer()
    import torch

    device = _model._device  # type: ignore[attr-defined]
    cfg = _cfg
    feature_list = list(cfg.feature_list)
    time_feature_list = list(cfg.time_feature_list)

    df = history.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
    df = df.sort_values("timestamp").reset_index(drop=True)
    if "volume" in df.columns and "vol" not in df.columns:
        df["vol"] = df["volume"]
    if "amt" not in df.columns:
        df["amt"] = df["vol"] * df["close"]
    if "outstanding_share" not in df.columns:
        df["outstanding_share"] = 0.0
    ts = df["timestamp"]
    df["minute"] = ts.dt.minute
    df["hour"] = ts.dt.hour
    df["weekday"] = ts.dt.weekday
    df["day"] = ts.dt.day
    df["month"] = ts.dt.month
    df = df.set_index("timestamp")
    df.index.name = "datetime"

    ad = pd.Timestamp(as_of_date)
    if ad not in df.index or df.index.get_loc(ad) < FUSION_L:
        return 0.0, 0.0
    idx = df.index.get_loc(ad)
    win = df.iloc[idx - FUSION_L: idx + 1]
    feat = win[feature_list].values.astype(np.float64)
    past = feat[:FUSION_L]
    mu = past.mean(0); sd = past.std(0)
    x = np.clip((feat - mu) / (sd + 1e-5), -FUSION_CLIP, FUSION_CLIP)
    x_ctx = x[:FUSION_L]
    stamp = win[time_feature_list].values.astype(np.float64)[:FUSION_L]
    dates = [d.strftime("%Y-%m-%d") for d in win.index]
    tlookup = _get_text_lookup(symbol)
    txt = np.zeros((FUSION_L, _text_encoder.dim_out), dtype=np.float32)
    for i in range(FUSION_L):
        e = tlookup.get(dates[i])
        if e is not None:
            txt[i] = e
    xs = torch.tensor(np.stack([x_ctx]), dtype=torch.float32, device=device)
    ss = torch.tensor(np.stack([stamp]), dtype=torch.float32, device=device)
    ts_t = torch.tensor(np.stack([txt]), dtype=torch.float32, device=device)
    t0, t1 = _tokenizer.encode(xs, half=True)
    with torch.no_grad():
        ret = _model.predict_return(t0, t1, ss, text_emb=ts_t)
    pred_ret_pct = float(ret.detach().cpu().numpy()[0])
    pred_ret_frac = pred_ret_pct / RETURN_TARGET_SCALE
    _scale = scale if scale is not None else FUSION_DIRECTION_SCALE
    direction = max(-1.0, min(1.0, pred_ret_frac / _scale)) if _scale > 0 else 0.0
    return direction, pred_ret_frac
