# 前期融合（Early Fusion）设计文档

把文本嵌入当作**额外 token** 拼进 predictor 输入，让 Transformer 在同一条序列内对
「价格 token」与「文本 token」做跨模态联合注意力，从而用文本直接条件化价格预测。

> 状态：**脚手架原型**。已跑通 CPU 端到端通路并通过结构正确性验证，
> 但**不追求训练质量**——真实前期融合需要覆盖训练期的文本语料与 GPU 算力。

---

## 1. 与后期融合的区别

| | 前期融合（本方案） | 后期融合 / LLM orchestrator |
|---|---|---|
| 融合位置 | 文本 token 进入 predictor 输入序列 | 在 Kronos 输出之后，用 LLM 读预测+新闻出结论 |
| 交互深度 | 深层：每层自注意力都跨模态 | 浅层：只在结果层结合 |
| 代价 | **重**：需重训 predictor，需时间对齐数据 | 轻：无需重训，但文本无法影响预测过程 |
| 数据要求 | 价格—文本按日对齐（稀缺） | 只需预测时点的文本 |

用户选型：**FinBERT 生成嵌入 + 因子文本与外部新闻两者结合 + 前期融合**。

> 注：「用通用 LLM 当 orchestrator 读 Kronos 预测+新闻出结论」属于后期融合，
> 不是本方案的实现目标；本方案保留的是前期融合能力。

---

## 2. 架构设计

### 2.1 文本 token 通道

在预训练 `Kronos` 之上包一层 `KronosFusion`（`model/kronos_fusion.py`）：

```
backbone  = 预训练 Kronos（权重原样加载，保留原预测能力）
text_proj = Linear(text_dim -> d_model)      # 新增：文本嵌入投影
modality_emb[2]                              # 新增：模态标识（0=文本, 1=价格）
null_text (text_dim)                         # 新增：未来日期无文本的占位向量
```

融合后的前向：

```
price_emb = HierarchicalEmbedding(s1,s2) + TemporalEmbedding(stamp)
text_emb  = text_proj(text) + TemporalEmbedding(同一天)   # 文本与同日价格共享时间嵌入
fused     = interleave(price_emb, text_emb)               # 见下
fused     -> N 层 Transformer（跨模态联合注意力）
logits    = head(fused 的价格位置)                        # 只在价格位置算 loss
```

### 2.2 为什么必须「交错」而不是「前置」（关键）

Kronos 的自注意力是 `is_causal=True`（`MultiHeadAttentionWithRoPE`）。

- **prepend** `[t_0..t_{T-1}, p_0..p_{T-1}]`：
  `p_t` 位于 `T+t`，能 attend 到位置 `< T+t`，即**能看到 t 日之后的文本 → 未来泄漏**。
- **interleave** `[t_0,p_0,t_1,p_1,...]`（默认）：
  `p_t` 位于 `2t+1`，只能 attend 到 `≤2t+1`，恰好覆盖 `text_0..text_t`，
  **不会看到未来日期的文本**，因果正确。

`tests/test_kronos_fusion.py` 对这两点做了断言验证（interleave 无泄漏 / prepend 有泄漏）。

### 2.3 训练目标

不变：仍用 `DualHead.compute_loss` 预测下一个价格 token（`s1`/`s2`）。
文本只作为**条件**输入，不参与预测目标，因此可直接复用原损失与预训练权重。

---

## 3. 数据流

```
sequoia_v2.db
 ├─ 因子文本  factor_dragon_tiger.reason、factor_theme.reason/tags   ─┐
 └─ 外部新闻  news(symbol, date, title, content)  【可插拔，缺失则跳过】─┤
                                                                      ▼
                              text_source.build_text_by_symbol()  → {symbol: {date: text}}
                                                                      ▼
                              text_encoder（FinBERT 或 hash 兜底）→ 句向量
                                                                      ▼
                         build_text_embeddings.py → text_embeddings.pkl（缓存）
                                                                      ▼
        QlibFusionDataset.__getitem__ → (x, x_stamp, text[window, text_dim])
                                                                      ▼
                              KronosFusion.forward(..., text_emb=text)
```

**未来泄漏控制**：因子表带 `announce_date`，披露日晚于所属交易日的行会被丢弃
（当日不可知的信息不能用于当日预测）。

---

## 4. 文件清单

| 文件 | 作用 |
|------|------|
| `model/kronos_fusion.py` | `KronosFusion` 模型 + `build_fusion_from_pretrained` + 融合推理 |
| `finetune/text_source.py` | 因子文本 + 可插拔新闻 → 按 symbol+date 对齐的文本 |
| `finetune/text_encoder.py` | FinBERT（真实）/ hash（离线兜底）编码器 |
| `finetune/build_text_embeddings.py` | 文本编码并缓存为 `text_embeddings.pkl` |
| `finetune/dataset_fusion.py` | `QlibFusionDataset`，额外返回逐日对齐文本 |
| `finetune/train_predictor_fusion.py` | 融合 predictor 训练脚本 |
| `finetune/config_sequoia_fusion.py` | 融合配置（与纯价格流程隔离） |
| `run_sequoia_fusion_cpu.sh` | CPU 一键冒烟 |
| `tests/test_kronos_fusion.py` | 结构正确性验证（含因果性） |

---

## 5. 使用方法

### 5.1 CPU 冒烟（离线兜底嵌入）

```bash
bash run_sequoia_fusion_cpu.sh
```

### 5.2 用真实 FinBERT

```bash
pip install transformers          # 已写入 requirements.txt
TEXT_ENCODER_MODE=auto bash run_sequoia_fusion_cpu.sh
# 或强制：TEXT_ENCODER_MODE=finbert
```

`auto` = 优先 FinBERT，不可用时自动降级为 hash（日志会明确提示）。

### 5.3 接入外部新闻（"两者结合"）

在 `data/sequoia_v2.db` 建表即可，无需改代码：

```sql
CREATE TABLE news (
    symbol TEXT,
    date   TEXT,      -- YYYY-MM-DD
    title  TEXT,
    content TEXT
);
```

`text_source` 会自动读取并与因子文本合并（同一天多条文本拼接）。

### 5.4 可调参数（环境变量）

| 变量 | 默认 | 说明 |
|------|------|------|
| `TEXT_ENCODER_MODE` | `hash`（脚本内）/ `auto`（配置内） | `auto`/`finbert`/`hash` |
| `TEXT_FINBERT_MODEL` | `ProsusAI/finbert` | 可换 FinGPT 等 |
| `FUSION_MODE` | `interleave` | `interleave`(因果正确)/`prepend`(消融) |
| `SEQUOIA_CPU_MAX_SYMBOLS` | `20` | 标的数量 |
| `FUSION_EPOCHS` / `FUSION_TRAIN_ITER` | `1` / `200` | CPU 冒烟预算 |
| `TRAIN_RANGE_START/END`、`VAL_RANGE_START/END` | 原默认区间 | 日期区间（见 §6.1） |

---

## 6. 已知限制与重要发现

### 6.1 ⚠ 文本与训练期不重叠（最关键）

实测发现：

- 因子文本（`factor_dragon_tiger` / `factor_theme`）**只覆盖 2026-05-07 ~ 2026-09-11**（约 90 个交易日）；
- 而默认训练/验证价格区间是 **2024-01-01 ~ 2025-06-30 / 2025-07-01 ~ 2026-02-28**，**与文本期完全不重叠**。

直接用默认区间跑，文本覆盖率会是 **0%**（等于没融合）。因此 `sequoia_db_loader.py`
的日期区间已改为支持环境变量覆盖，融合冒烟脚本把区间指到与文本重叠的
`2026-02-01 ~ 2026-09-11`。

> 该区间的 train/val **有重叠**，仅用于验证通路，不用于评估模型质量。

### 6.2 文本天然稀疏

即使区间对齐，覆盖率仍只有 **3.2%**（95 个有文本的日 / 2944 行）：
龙虎榜与题材是**事件驱动**的，并非每天都有。无文本的日期用零向量表示
（语义正确：当日无消息）。要提升信息量，必须接入**每日新闻语料**。

### 6.3 90 个交易日 < 101 行窗口

单个训练窗口需 `lookback(90) + predict(10) + 1 = 101` 行，
而文本期仅约 90 个交易日 —— **当前语料无法支撑严谨的 train/val 划分**。
这是「价格—文本对齐数据稀缺」的直接体现，也是真实的工程约束。

### 6.4 融合推理返回全长序列

`auto_regressive_inference_fused` 与官方 `auto_regressive_inference` 约定一致：
返回 `[B, T_ctx + pred_len, D]`，调用方需自行 `preds[:, -pred_len:, :]` 取预测段。

### 6.5 训练算力

前期融合需**重训 predictor**，序列长度翻倍（价格+文本 token），
真实训练应在 GPU 上进行；CPU 仅够冒烟。

---

## 7. 验证结果

**CPU 冒烟**（20 标的 / 200 iter / 1 epoch）：

```
[fusion] text_dim=128, fusion_mode=interleave
Fusion Predictor Model Size: 24.8M
[fusion-dataset] 文本覆盖率: 95/2944 行 (3.2%)，text_dim=128
Validation Loss: 2.7443
Best model saved to ./outputs/models/sequoia_predictor_fusion/checkpoints/best_model
```

**结构正确性测试**（`python tests/test_kronos_fusion.py`）5/5 通过：

| 测试 | 验证内容 |
|------|---------|
| `test_output_shape_and_price_positions` | logits 只在价格位置输出 |
| `test_text_affects_prediction` | 文本确实进入计算、影响输出 |
| `test_interleave_no_future_text_leak` | **无未来文本泄漏（因果正确）** |
| `test_prepend_has_future_leak` | 对照：prepend 确实泄漏 |
| `test_gradient_flows_into_text_projection` | `text_proj` 能收到梯度 |

**融合推理**：checkpoint 可被 `KronosFusion.from_pretrained` 加载，
自回归生成正常（输出 `[1, T+pred_len, 7]`）。

---

## 8. 下一步

1. **接入覆盖训练期的每日新闻语料**（当前最大瓶颈）→ 覆盖率与信息量才能上来。
2. 在 GPU 上按 `config_sequoia.py` 全量重训融合 predictor。
3. 补充 `KronosPredictorFusion`（DataFrame 级融合推理封装，目前需自行切片与查表）。
4. 若要做「LLM orchestrator」，那是后期融合方向，与本模块正交，可另行设计。
