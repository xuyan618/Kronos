"""
eval_predictor_fusion.py —— 用训练好的融合 predictor 在验证集上做推理，
查看「实际预测分布」并给出信号质量指标。

做法：
  1. 加载 best_model（含训练好的文本通道权重）。
  2. 在验证集上以 teacher-forcing 跑前向，对 s1/s2 logits 取 argmax 得到预测 token。
  3. 用 tokenizer.decode 把 预测token / 实际target token 解码回特征空间，
     取 close 通道（z 空间）作为预测值代理。
  4. 统计预测分布 + IC / RankIC / 方向准确率。

注：特征是按窗口 z-score 归一，close 为「相对窗口均值的 z 分值」，
方向(正负)表示「当天相对近期均值是偏高/偏低」，可作为方向信号。
"""
import os
import sys

import numpy as np
import torch

sys.path.append('.')
from config_sequoia_fusion import get_config
from dataset_fusion import QlibFusionDataset
from model_factory import build_tokenizer
from torch.utils.data import DataLoader
from model.kronos_fusion import build_fusion_from_pretrained, KronosFusion
from safetensors.torch import load_file

CLOSE_IDX = 3  # feature_list = [open, high, low, close, vol, amt, outstanding_share]


def rank_ic(a, b):
    a_r = a.argsort().argsort().astype(float)
    b_r = b.argsort().argsort().astype(float)
    return np.corrcoef(a_r, b_r)[0, 1]


def main():
    config_instance = get_config()
    config = config_instance.__dict__
    device = torch.device('cpu')

    best_model = os.path.join(
        config['save_path'], config['predictor_save_folder_name'], 'checkpoints', 'best_model'
    )
    assert os.path.exists(best_model), f"best_model 不存在: {best_model}"
    print(f"[eval] best_model: {best_model}")

    # ---- 加载模型（含训练好的文本通道）----
    print("[eval] 加载融合模型...")
    model = None
    try:
        model = KronosFusion.from_pretrained(best_model)
        print("  通过 KronosFusion.from_pretrained 加载完整权重")
    except Exception as e:  # noqa
        print(f"  from_pretrained 失败({e})，回退到 build + 加载 safetensors")
        model = build_fusion_from_pretrained(
            config['pretrained_predictor_path'],
            text_dim=int(config.get('text_dim', 768)),
            fusion_mode=config.get('fusion_mode', 'interleave'),
            use_modality_emb=bool(config.get('use_modality_emb', True)),
            device=device,
        )
        sd = load_file(os.path.join(best_model, 'model.safetensors'))
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"  missing={len(missing)} unexpected={len(unexpected)}")
    model = model.to(device).eval()

    # ---- tokenizer（finetuned）----
    tokenizer = build_tokenizer(config, finetuned=True).to(device).eval()

    # ---- 验证集 ----
    val_dataset = QlibFusionDataset('val')
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    print(f"[eval] 验证集样本数: {len(val_dataset)}，推理前 {min(150, len(val_loader))} 个 batch（约 {min(150, len(val_loader)) * 16} 样本）")

    preds, acts = [], []
    n_batches = 0
    with torch.no_grad():
        for batch_x, batch_x_stamp, batch_text in val_loader:
            n_batches += 1
            if n_batches > 150:
                break
            batch_x = batch_x.to(device)
            batch_x_stamp = batch_x_stamp.to(device)
            batch_text = batch_text.to(device)

            token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)
            s1_in, s2_in = token_seq_0[:, :-1], token_seq_1[:, :-1]
            stamp_in = batch_x_stamp[:, :-1, :]
            text_in = batch_text[:, :-1, :]

            s1_logits, s2_logits = model(s1_in, s2_in, stamp_in, text_emb=text_in)
            pred_s1 = s1_logits.argmax(-1)
            pred_s2 = s2_logits.argmax(-1)

            pred_feat = tokenizer.decode([pred_s1, pred_s2], half=True)   # [B, T-1, d_in]
            act_s1 = token_seq_0[:, 1:]
            act_s2 = token_seq_1[:, 1:]
            act_feat = tokenizer.decode([act_s1, act_s2], half=True)

            preds.append(pred_feat[..., CLOSE_IDX].detach().cpu().numpy())
            acts.append(act_feat[..., CLOSE_IDX].detach().cpu().numpy())

    pred = np.concatenate(preds).ravel()
    act = np.concatenate(acts).ravel()
    print(f"[eval] 收集到 {pred.size} 个预测点")

    # ---- 指标 ----
    def stats(x):
        return (f"mean={x.mean():.3f} std={x.std():.3f} "
                f"p5={np.percentile(x,5):.3f} p25={np.percentile(x,25):.3f} "
                f"p50={np.percentile(x,50):.3f} p75={np.percentile(x,75):.3f} "
                f"p95={np.percentile(x,95):.3f}")

    ic = np.corrcoef(pred, act)[0, 1]
    ric = rank_ic(pred, act)
    da = np.mean(np.sign(pred) == np.sign(act))

    long_act = act[pred > 0].mean() if (pred > 0).any() else float('nan')
    short_act = act[pred < 0].mean() if (pred < 0).any() else float('nan')

    print("\n==================== 验证集推理结果 ====================")
    print(f"预测点数量          : {pred.size}")
    print(f"预测 close(z) 分布  : {stats(pred)}")
    print(f"实际 close(z) 分布  : {stats(act)}")
    print(f"IC (Pearson)        : {ic:.4f}")
    print(f"RankIC (Spearman)   : {ric:.4f}")
    print(f"方向准确率(符号一致): {da:.4f}  (随机基准 0.50)")
    print(f"pred>0 时 实际均值  : {long_act:.4f}")
    print(f"pred<0 时 实际均值  : {short_act:.4f}")
    print("=======================================================")
    print("说明: close 为逐窗口 z-score 归一后的分值；方向(正负)表示相对窗口均值的偏高/偏低。")
    print("      IC/RankIC 衡量预测与实际的秩相关，方向准确率为符号一致比例。")


if __name__ == '__main__':
    main()
