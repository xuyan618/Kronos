"""
backtest_predictor_fusion.py —— 融合模型「近期区间」回测（自回归次日预测）。

做法：
  对每只股票最靠后的交易日 t：
    - 取上下文窗口 [t-L, t-1]（L=90），按训练方式 z-score 归一；
    - 用模型自回归 decode（decode_s1/decode_s2 + argmax，确定性）预测 t 日 close 的 z 分值（方向信号）；
    - 实际次日收益用原始 close：close[t]/close[t-1]-1。
  一次推理收集预测后，汇报多个「回测窗口长度」× 多个「TOP_FRAC」下的：
    - 日度 RankIC 均值 / IC>0 占比；
    - 多头(前Frac等权)、多空(前Frac-后Frac)、等权基准 的总收益与净值。

说明：
  本数据 train/val/test 为同一段时期（2026-01-05~2026-09-10）的非时间随机切分，
  故属「近期区间预测评估」而非严格时间外；自回归预测不泄露 t 日真实 token，仍衡量预测能力。
  严格时间外需按日期重切并重训（见任务1）。
"""
import os
import sys
import pickle

import numpy as np
import torch

sys.path.append('.')
from config_sequoia_fusion import get_config
from model_factory import build_tokenizer
from model.kronos_fusion import build_fusion_from_pretrained, KronosFusion
from safetensors.torch import load_file

CLOSE_IDX = 3          # open/high/low/close/vol/amt/outstanding_share
L = 90                 # lookback（与训练一致）
MAX_WINDOW = int(os.environ.get("MAX_WINDOW", "120"))   # 收集预测的天数
WINDOWS = [40, 120]    # 汇报的回测窗口长度（取收集区间的最后 N 天）
TOP_FRACS = [0.10, 0.20, 0.30]   # 多头/多空占比
CHUNK = 48             # 每批处理的标的数量


def rank_ic(a, b):
    a_r = a.argsort().argsort().astype(float)
    b_r = b.argsort().argsort().astype(float)
    return np.corrcoef(a_r, b_r)[0, 1]


# 强制确定性推理：CPU 多线程下矩阵乘的归约顺序会导致弱信号预测的 argmax 在运行间抖动，
# 从而使截面 IC 在 ±0.08 间不稳定。固定线程数+种子+确定性算法以消除该噪声。
import torch as _torch
_torch.manual_seed(0)
np.random.seed(0)
# 仅在确定性模式(SEQUOIA_DETERMINISTIC=1)下钉死单线程，避免弱信号预测抖动；
# 否则放开多线程，充分利用多核 CPU 加速回测。
if os.environ.get("SEQUOIA_DETERMINISTIC") == "1":
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    _torch.set_num_threads(1)
    try:
        _torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def compute_metrics(preds, acts, top_frac):
    daily_ic, long_r, ls_r, mkt_r = [], [], [], []
    for p, a in zip(preds, acts):
        if len(p) < 5:
            continue
        ric = rank_ic(p, a)
        if np.isfinite(ric):
            daily_ic.append(ric)
        order = np.argsort(-p)
        n = len(order)
        k = max(1, int(round(n * top_frac)))
        top = order[:k]
        bot = order[n - k:]
        lr = a[top].mean()
        sr = a[bot].mean()
        long_r.append(lr)
        ls_r.append(lr - sr)
        mkt_r.append(a.mean())
    nav_long = np.cumprod(1 + np.array(long_r))
    nav_ls = np.cumprod(1 + np.array(ls_r))
    nav_mkt = np.cumprod(1 + np.array(mkt_r))
    ic = np.array(daily_ic)
    return dict(
        n_days=len(long_r),
        ic_mean=ic.mean() if len(ic) else float('nan'),
        ic_pos=(ic > 0).mean() if len(ic) else float('nan'),
        long_total=nav_long[-1] - 1 if len(nav_long) else float('nan'),
        ls_total=nav_ls[-1] - 1 if len(nav_ls) else float('nan'),
        mkt_total=nav_mkt[-1] - 1 if len(nav_mkt) else float('nan'),
        nav_long=nav_long, nav_ls=nav_ls, nav_mkt=nav_mkt,
    )


def main():
    config_instance = get_config()
    config = config_instance.__dict__
    device = torch.device('cpu')

    base = config['dataset_path']
    best_model = os.path.join(
        config['save_path'], config['predictor_save_folder_name'], 'checkpoints', 'best_model'
    )
    assert os.path.exists(best_model), f"best_model 不存在: {best_model}"

    print("[bt] 加载融合模型...")
    try:
        model = KronosFusion.from_pretrained(best_model)
    except Exception as e:  # noqa
        print(f"  from_pretrained 失败({e})，回退")
        model = build_fusion_from_pretrained(
            config['pretrained_predictor_path'],
            text_dim=int(config.get('text_dim', 768)),
            fusion_mode=config.get('fusion_mode', 'interleave'),
            use_modality_emb=bool(config.get('use_modality_emb', True)),
            device=device,
        )
        sd = load_file(os.path.join(best_model, 'model.safetensors'))
        model.load_state_dict(sd, strict=False)
    model = model.to(device).eval()
    tokenizer = build_tokenizer(config, finetuned=True).to(device).eval()
    feature_list = config_instance.feature_list
    time_feature_list = config_instance.time_feature_list
    print(f"[bt] 特征={feature_list}  时间特征={time_feature_list}")

    print("[bt] 读取测试数据...")
    with open(os.path.join(base, 'test_data.pkl'), 'rb') as f:
        raw = pickle.load(f)
    with open(os.path.join(base, 'text_embeddings.pkl'), 'rb') as f:
        tpack = pickle.load(f)
    tdata = tpack.get('data', {})

    # 复刻 QlibDataset 的时间特征生成（pickle 仅含价格列）
    for s in list(raw.keys()):
        df = raw[s].reset_index()
        dt = df['datetime']
        df['minute'] = dt.dt.minute
        df['hour'] = dt.dt.hour
        df['weekday'] = dt.dt.weekday
        df['day'] = dt.dt.day
        df['month'] = dt.dt.month
        raw[s] = df.set_index('datetime')[feature_list + time_feature_list]

    def build_text_lookup(symbol):
        sp = tdata.get(symbol)
        if sp is None:
            return {}
        return {d: e for d, e in zip(sp['dates'], sp['emb'])}

    all_dates = set()
    for s, df in raw.items():
        all_dates.update(df.index)
    all_dates = sorted(all_dates)
    bt_dates = all_dates[-MAX_WINDOW:]
    print(f"[bt] 收集区间: {bt_dates[0].date()} ~ {bt_dates[-1].date()} （{len(bt_dates)} 交易日）")

    symbols = list(raw.keys())
    dates_daily, pred_daily, act_daily = [], [], []

    with torch.no_grad():
        for di, dt in enumerate(bt_dates):
            dstr = dt.strftime('%Y-%m-%d')
            batch_x, batch_stamp, batch_text, batch_aret = [], [], [], []
            for s in symbols:
                df = raw[s]
                if dt not in df.index:
                    continue
                idx = df.index.get_loc(dt)
                if idx < L:
                    continue
                win = df.iloc[idx - L: idx + 1]
                feat = win[feature_list].values.astype(np.float64)
                past = feat[:L]
                mu = past.mean(0)
                sd = past.std(0)
                x = (feat - mu) / (sd + 1e-5)
                x = np.clip(x, -5.0, 5.0)
                x_ctx = x[:L]
                stamp = win[time_feature_list].values.astype(np.float64)[:L]
                tl = build_text_lookup(s)
                txt = np.zeros((L, tpack.get('dim', 768)), dtype=np.float32)
                for i in range(L):
                    e = tl.get(win.index[i].strftime('%Y-%m-%d'))
                    if e is not None:
                        txt[i] = e
                close_raw = df['close'].values
                aret = close_raw[idx] / close_raw[idx - 1] - 1.0
                if not np.isfinite(aret):
                    continue
                batch_x.append(x_ctx)
                batch_stamp.append(stamp)
                batch_text.append(txt)
                batch_aret.append(aret)

            if len(batch_aret) < 5:
                continue

            pred_sig = np.empty(len(batch_aret), dtype=np.float64)
            pos = 0
            while pos < len(batch_aret):
                end = min(pos + CHUNK, len(batch_aret))
                xs = torch.tensor(np.stack(batch_x[pos:end]), dtype=torch.float32, device=device)
                ss = torch.tensor(np.stack(batch_stamp[pos:end]), dtype=torch.float32, device=device)
                ts = torch.tensor(np.stack(batch_text[pos:end]), dtype=torch.float32, device=device)
                t0, t1 = tokenizer.encode(xs, half=True)
                s1_logits, price_ctx = model.decode_s1(t0, t1, ss, text_emb=ts)
                ps1 = s1_logits[:, -1, :].argmax(-1)
                ps1_seq = ps1.unsqueeze(1)
                s2_logits = model.decode_s2(price_ctx, ps1_seq)
                ps2 = s2_logits[:, -1, :].argmax(-1)
                feat_dec = tokenizer.decode([ps1_seq, ps2.unsqueeze(1)], half=True)
                pred_sig[pos:end] = feat_dec[:, 0, CLOSE_IDX].detach().cpu().numpy()
                pos = end

            dates_daily.append(dt)
            pred_daily.append(pred_sig)
            act_daily.append(np.array(batch_aret, dtype=np.float64))
            if di % 20 == 0 or di == len(bt_dates) - 1:
                ric = rank_ic(pred_sig, act_daily[-1])
                print(f"  {dstr}: n={len(batch_aret)} rankIC={ric:.3f}")

    # ---- 多窗口 × 多阈值 汇报 ----
    print("\n==================== 融合模型回测：窗口×阈值对比 ====================")
    print(f"{'窗口':>6} {'TOP%':>5} {'RankIC':>8} {'IC>0':>6} {'多头':>8} {'多空':>8} {'基准':>8}")
    saved = None
    for W in WINDOWS:
        for tf in TOP_FRACS:
            p = pred_daily[-W:]
            a = act_daily[-W:]
            m = compute_metrics(p, a, tf)
            print(f"{W:>6} {int(tf*100):>4}% {m['ic_mean']:>8.4f} {m['ic_pos']:>6.2%} "
                  f"{m['long_total']:>8.2%} {m['ls_total']:>8.2%} {m['mkt_total']:>8.2%}")
            if W == 120 and abs(tf - 0.20) < 1e-9:
                saved = m
    print("======================================================================")
    print("说明: 信号=模型自回归预测 t 日 close 的 z 分值(方向代理)；多头=前Frac等权；")
    print("      多空=前Frac-后Frac；基准=全市场等权。数据为 2026 同段非时间切分，属近期区间评估。")

    out_dir = os.path.join(config['save_path'], 'sequoia_predictor_fusion', 'backtest')
    os.makedirs(out_dir, exist_ok=True)
    if saved is not None:
        np.savez(os.path.join(out_dir, 'nav.npz'),
                 dates=np.array([d.strftime('%Y-%m-%d') for d in dates_daily[-120:]]),
                 nav_long=saved['nav_long'], nav_ls=saved['nav_ls'], nav_mkt=saved['nav_mkt'])
        print(f"[bt] 净值(窗口120/TOP20%)已保存: {out_dir}/nav.npz")
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            dts = dates_daily[-120:]
            plt.figure(figsize=(10, 5))
            plt.plot(dts, saved['nav_long'], label=f"LongTop20% ({saved['long_total']:.1%})")
            plt.plot(dts, saved['nav_ls'], label=f"LongShort ({saved['ls_total']:.1%})")
            plt.plot(dts, saved['nav_mkt'], label=f"Market ({saved['mkt_total']:.1%})", linestyle='--')
            plt.title('KronosFusion Backtest NAV (recent-period)')
            plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
            plt.savefig(os.path.join(out_dir, 'nav.png'), dpi=150)
            print(f"[bt] 净值图: {out_dir}/nav.png")
        except Exception as e:
            print(f"[bt] 绘图跳过: {e}")


if __name__ == '__main__':
    main()
