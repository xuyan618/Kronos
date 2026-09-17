# Kronos 融合模型 训练记录 (Training Log)

> 文档版本：v1
> 目的：每次训练/回测后，把「模式 + 超参 + 结果 + 已知问题」追加到本文档，方便后期回溯与横向对比。
> 格式：先更新顶部「汇总表」，再在下方写一条 `## Run N` 明细。复制模板即可。

---

## 0. 背景：两套融合方案并存

| 方案 | 训练/推理 | 入口 | 是否参与 bat 训练 |
|---|---|---|---|
| **早期融合 (Early Fusion)** | 训练期，模型内部 | `finetune/train_predictor_fusion.py` → `model/kronos_fusion.py` (`KronosFusion`) | ✅ 是（bat 真正训练的） |
| **决策层融合 (Decision-layer)** | 推理期，规则加权 | `app/decision/fusion.py` → `generate_trade_signal` | ❌ 否（0.6/0.4 规则，用冻结 FinBERT 实时打分，仅 `app/pipeline.py` 调用） |

> 本文档记录的是 **早期融合** 的训练结果（bat 跑出来的）。决策层融合当前未独立训练，不在对比范围内。

## 汇总表

| Run | 日期 | 文本编码器 | fusion_mode | 文本来源 | 文本覆盖 | RankIC | IC>0 | 多头(20%) | 多空(20%) | 基准 | 最佳ValLoss | 状态 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2026-09-17 | yiyanghkust/finbert-tone-chinese (768) | interleave | factor+news | 3.0% | **-0.0815** | 25% | -7.80% | -8.41% | +0.10% | 3.0784 (ep1) | ⚠️ 信号无效/过拟合 |
| 2 | 2026-09-17 | 同Run1 + A/B/C/D | interleave | factor+news(±3d) | TBD | TBD | TBD | TBD | TBD | TBD | TBD | 🔄 改进中 |

---

## Run 1 — 基线 (Baseline)

- **日期**：2026-09-17
- **执行方式**：`run_sequoia_fusion_temporal_gpu.bat`（Windows, RTX 3060, CUDA）
- **融合方案**：早期融合 `KronosFusion`（模型内部融合价格 + FinBERT 文本嵌入）

### 配置 (Config)
| 项 | 值 | 来源/备注 |
|---|---|---|
| 文本编码器 | `TEXT_ENCODER_MODE=auto` → `yiyanghkust/finbert-tone-chinese` | dim=768, device=cuda |
| FinBERT 模型 | `TEXT_FINBERT_MODEL=yiyanghkust/finbert-tone-chinese` | 中文金融 FinBERT |
| 文本来源 | `text_sources=("factor","news")` | factor_dragon_tiger / factor_theme / news |
| 融合布局 | `FUSION_MODE=interleave` | 因果正确 [t0,p0,t1,p1,...] |
| 训练样本/epoch | `FUSION_TRAIN_ITER=5000`, `FUSION_VAL_ITER=200` | |
| Epochs | `FUSION_EPOCHS=10` | |
| Batch | `FUSION_BATCH_SIZE=64`, `FUSION_NUM_WORKERS=4` | |
| 学习率 | `predictor_learning_rate=5e-4`（继承 base，bat 未覆盖） | OneCycleLR, pct_start=0.03, div_factor=10 |
| 梯度裁剪 | `clip_grad_norm=3.0`（硬编码） | |
| 标的数量 | `SEQUOIA_CPU_MAX_SYMBOLS=300` | 可用标的 300/5215 |
| 时间切分 | train 2024-01-02~2025-09-30 / val 2025-10-01~2026-03-31 / test 2026-04-01~2026-09-14 | 严格时序 |
| Tokenizer | **回退 base 扩展版**（缺 `sequoia_tokenizer`，`d_in_new=1` 随机初始化冻结） | 价格分词器，非文本 |
| 模型规模 | 103.0M 参数 | |

### 数据 & 文本对齐
- 行情原始行：3,338,250；可用标的(≥101行)：300
- 文本条数：factor_dragon_tiger 55,663 / factor_theme 52,622 / news 59,549
- 对齐出 (symbol,date) 对：**8,790**（285 只标的）
- **融合数据集文本覆盖率：train 3.0% (3873/127289) / val 3.8% (1303/34742)** ← 关键瓶颈

### 训练过程
| Epoch | Val Loss | Train Loss(抽样) | 备注 |
|---|---|---|---|
| 1 | **3.0784** (best, saved) | 2.63 | 最佳 checkpoint |
| 2 | 3.1539 | 2.32 | |
| 3 | 3.1806 | 2.50 | |
| 4 | 3.1664 | 2.18 | |
| 5 | 3.1939 | — | |
| 6 | 3.2029 | 2.18 | |
| 7 | 3.2256 | 2.18 | |
| 8 | 3.2325 | 2.23 | |
| 9 | 3.2322 | 2.19 | |
| 10 | 3.2304 | — | |

- **总训练时长 ≈ 3h37m**
- **过拟合明显**：Train loss 降、Val loss 从 ep1 起单调升 → 泛化变差，最佳模型停在 ep1。

### 回测结果 (Test: 2026-04-01~2026-09-14, 114 交易日, 严格 OOS)

| 窗口 | TOP% | RankIC | IC>0 | 多头 | 多空 | 基准 |
|---|---|---|---|---|---|---|
| 40 | 10% | -0.0815 | 25% | -11.87% | -11.22% | +0.10% |
| 40 | 20% | -0.0815 | 25% | -7.80% | -8.41% | +0.10% |
| 40 | 30% | -0.0815 | 25% | -6.26% | -8.02% | +0.10% |
| 120 | 10% | -0.0815 | 25% | -11.87% | -11.22% | +0.10% |
| 120 | 20% | -0.0815 | 25% | -7.80% | -8.41% | +0.10% |
| 120 | 30% | -0.0815 | 25% | -6.26% | -8.02% | +0.10% |

- 输出：`outputs/models/sequoia_predictor_fusion/backtest/{nav.npz, nav.png}`
- **信号无效**：RankIC 为负且极弱，IC>0 仅 25%（差于随机），多头/多空全负，基准近乎持平。

### 已知问题 / TODO
1. ⚠️ **回测窗口 bug**：`WINDOWS=[40,120]` 未生效，40 与 120 结果完全一致（`compute_metrics` 未对最后 N 天截取）。需修复才能做真实窗口对比。
2. ⚠️ **信号可能反向**：RankIC=-0.08 疑似符号反了（模型预测 close 的 z 分 vs 回测用收益率）。建议把 `preds` 取负重算验证；若变 +0.08，则仅符号问题，决策层加翻转即可。
3. ⚠️ **文本覆盖仅 3%**：融合基本≈纯价格模型，FinBERT 一路几乎为 0。需查 `dataset_fusion.py` 的 (symbol,date) 对齐逻辑（放宽 ±1 天 / 规范 date 字段）。
4. ⚠️ **过拟合**：best 停在 ep1。可降 `FUSION_EPOCHS`、加 `weight_decay`、或冻结 backbone 只训 text_proj+解码头。
5. ℹ️ Tokenizer 回退 base 扩展版属预期（bat 不含 tokenizer 微调阶段），非错误。

### 结论
管线已端到端跑通（训练+回测产出结果），但**当前信号无预测价值**。根因为文本对齐率过低（融合形同虚设）+ 过拟合 + 疑似符号反向。下一步优先做「符号检查(A) + 回测窗口修复(B)」，再决定是否投入文本对齐大改(C)。

---

## Run 2 — 改进（A/B/C/D 已应用，待运行填结果）

- **日期**：2026-09-17（与 Run 1 同日，代码改进后重跑）
- **相对 Run 1 的改动**：
  - **A 符号检查**：回测新增「信号取负对照表」，直接看 RankIC 是否翻正。
  - **B 窗口修复**：回测改读 train+val+test 拼接完整历史，仅对测试区间发预测 → 40/120 窗口天数不同（输出新增「天数」列验证）。
  - **C 文本对齐**：`dataset_fusion.py` 日期匹配放宽到 ±3 天 → 文本覆盖率提升（日志打印新覆盖率）。
  - **D 过拟合**：`FUSION_WEIGHT_DECAY=0.01` + `FREEZE_BACKBONE=1`（冻结 transformer/embedding/time_emb，仅训文本通道+head+dep_layer+norm）。
- **待填结果**：RankIC / IC>0 / 多头 / 多空 / 基准 / 最佳ValLoss / 文本覆盖率。

> 运行：保持 `run_sequoia_fusion_temporal_gpu.bat` 默认（已含上述开关），跑完把结果贴回此处。

---

## 模板（复制此块新增 Run）

```markdown
## Run N — <一句话描述>

- **日期**：YYYY-MM-DD
- **执行方式**：`run_sequoia_fusion_temporal_gpu.bat` / 其他
- **融合方案**：早期融合 / 决策层 / 其他

### 配置 (Config)
| 项 | 值 | 备注 |
|---|---|---|
| 文本编码器 | | |
| 融合布局 | | |
| Epochs / Iter | | |
| Batch / LR | | |
| 标的数量 | | |
| 时间切分 | | |
| Tokenizer | | |

### 数据 & 文本对齐
- 文本覆盖率：train / val

### 训练过程
- 总时长：
- Val Loss 曲线：ep1= / epN= （best=）
- 过拟合? Y/N

### 回测结果 (Test 区间)
| 窗口 | TOP% | RankIC | IC>0 | 多头 | 多空 | 基准 |
|---|---|---|---|---|---|---|
| | | | | | | |

### 已知问题 / TODO
-

### 结论
-
```
