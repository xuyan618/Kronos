# Kronos 训练使用教程

本教程覆盖本仓库的**两条训练路线**：

| 路线 | 目标 | 脚本 | 状态 |
|------|------|------|------|
| **A. 扩展维度微调** | 在预训练权重上引入新因子维度（如 `outstanding_share`），7 维微调 | `run_sequoia_cpu.sh` / `run_sequoia_finetune.sh` | 已完成并验证 |
| **B. 前期融合** | 把新闻/因子文本嵌入当额外 token 拼进 predictor，跨模态联合预测 | `run_sequoia_fusion_cpu.sh` | 脚手架原型 |

> CPU 仅用于**流程验证与压测**；真实训练请在 GPU 机器上进行。

---

## 0. 环境准备

```bash
pip install -r requirements.txt
```

可选（按需）：

```bash
pip install transformers   # 前期融合用真实 FinBERT 嵌入时需要
pip install pytest         # 运行 tests/ 下测试时需要
```

数据：`data/sequoia_v2.db`（sqlite，含 `stock_daily` 行情与多张因子表）。

---

## 1. 路线 A：扩展维度微调

### 1.1 一键 CPU 冒烟（约 30 秒）

```bash
bash run_sequoia_cpu.sh
```

流程：生成价格 pickle → 微调扩展 tokenizer → 微调 predictor。

产出：
- `./outputs/models/sequoia_tokenizer/checkpoints/best_model`
- `./outputs/models/sequoia_predictor/checkpoints/best_model`

### 1.2 CPU 中等压测（120 标的 / 3 epoch，约 3 小时）

```bash
bash run_sequoia_cpu_medium.sh
```

### 1.3 自定义规模（不改代码）

`config_sequoia_cpu.py` 支持环境变量覆盖：

```bash
export SEQUOIA=1 SEQUOIA_CPU=1
export SEQUOIA_CPU_DATASET_PATH=./data/sequoia_cpu_200
export SEQUOIA_CPU_MAX_SYMBOLS=200     # 标的数量
export SEQUOIA_CPU_EPOCHS=5            # epoch 数
export SEQUOIA_CPU_TRAIN_ITER=1000000  # 很大 = 用满自然样本
export SEQUOIA_CPU_VAL_ITER=200000
export PYTHONPATH="$(pwd):$PYTHONPATH"

python -u finetune/sequoia_db_loader.py
python -u finetune/train_tokenizer.py
python -u finetune/train_predictor.py
```

### 1.4 GPU 全量训练

```bash
bash run_sequoia_finetune.sh      # 需 torchrun 多卡环境
```

### 1.5 CPU 路径原理（为什么不需要 torchrun）

`SEQUOIA_CPU=1` 时 `setup_ddp()` 走 CPU 单进程分支（`device='cpu'`、`backend='gloo'`、
`world_size=1`），所有 `dist.*` 调用均被 `if dist.is_initialized():` 守卫，
单进程直接 `python` 即可运行。详见 `finetune/CPU_FINETUNE_GUIDE.md`。

---

## 2. 路线 B：前期融合（文本 token）

### 2.1 完整流程

```bash
# ①（可选但强烈建议）抓取新闻入库
python finetune/fetch_news.py --limit 20 --pages 3 --page-size 50

# ② 一键冒烟：生成价格 → 构建文本嵌入 → 融合微调
bash run_sequoia_fusion_cpu.sh
```

产出：`./outputs/models/sequoia_predictor_fusion/checkpoints/best_model`

### 2.2 步骤①详解：新闻抓取入库

```bash
# 基本用法：抓取默认数据集里的前 20 只标的
python finetune/fetch_news.py --limit 20

# 指定标的 / 抓更多 / 限速
python finetune/fetch_news.py --symbols 000001,000002 --pages 5 --page-size 50 --sleep 0.5

# 只看不写
python finetune/fetch_news.py --limit 3 --dry-run
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `--limit` | `20` | 最多抓多少只标的（0=全部） |
| `--provider` | `eastmoney` | `eastmoney`(分页,无第三方依赖) / `akshare`(每只仅 10 条) |
| `--pages` | `3` | 每只标的抓几页 |
| `--page-size` | `50` | 每页条数 |
| `--sleep` | `0.3` | 请求间隔（限速，避免被封） |
| `--symbols` | — | 逗号分隔，优先于自动读取 |
| `--dry-run` | off | 只抓不写 |
| `--db` / `--dataset` | — | 指定数据库 / pickle 目录 |

**入库位置**：`data/sequoia_v2.db` 的 `news` 表

```sql
CREATE TABLE news (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL, date TEXT NOT NULL, title TEXT NOT NULL,
    content TEXT, source TEXT, url TEXT, created_at TEXT,
    UNIQUE(symbol, date, title)      -- 去重，重复运行增量累积
);
```

建好这张表后，`build_text_embeddings.py` 会**自动**把新闻与因子文本合并，无需改代码。
（若自己已有新闻数据，按上述表结构写入即可。）

### 2.3 文本编码器

```bash
pip install transformers
TEXT_ENCODER_MODE=auto bash run_sequoia_fusion_cpu.sh
```

| `TEXT_ENCODER_MODE` | 说明 |
|---------------------|------|
| `hash`（脚本默认） | 离线确定性哈希兜底，**无依赖无联网**，仅用于跑通通路，语义无意义 |
| `auto`（配置默认） | 优先 FinBERT，不可用时自动降级为 hash（日志会提示） |
| `finbert` | 强制 FinBERT（`ProsusAI/finbert`，[CLS] 768 维） |

### 2.4 融合参数

| 变量 | 默认 | 说明 |
|------|------|------|
| `FUSION_MODE` | `interleave` | **因果正确**；`prepend` 会泄漏未来文本，仅作消融对照 |
| `FUSION_EPOCHS` / `FUSION_TRAIN_ITER` | `1` / `200` | CPU 冒烟预算 |
| `TEXT_FINBERT_MODEL` | `ProsusAI/finbert` | 可换其他文本模型 |

---

## 3. 日期区间（重要）

因子文本与新闻都是**近期数据**，可能与默认训练期不重叠，导致文本覆盖率 0%。

`sequoia_db_loader.py` 的日期区间已支持环境变量覆盖（不设置则保持原值）：

```bash
export TRAIN_RANGE_START=2026-02-01 TRAIN_RANGE_END=2026-09-11
export VAL_RANGE_START=2026-03-01   VAL_RANGE_END=2026-09-11
export TEST_RANGE_START=2026-03-01  TEST_RANGE_END=2026-09-11
```

融合冒烟脚本已默认把这些区间指到与文本重叠的时段（见脚本内注释）。

---

## 4. 推理 / 预测

- **非融合**：`KronosPredictor`（`model/kronos.py`）的 `predict` / `predict_batch`。
- **融合**：`model/kronos_fusion.py` 的 `auto_regressive_inference_fused`，
  与官方 `auto_regressive_inference` 约定一致：**返回全长序列**
  `[B, T_ctx + pred_len, D]`，需自行切片取预测段：

```python
preds = auto_regressive_inference_fused(tok, model, x, x_stamp, y_stamp,
                                        text_emb, pred_len=P, ...)
preds = preds[:, -P:, :]     # 只保留预测段
```

未来日期没有文本，融合推理会自动使用模型内可学习的 `null_text` 占位。

---

## 5. 验证与测试

```bash
python tests/test_kronos_fusion.py      # 前期融合结构正确性（含因果性）
```

5 项测试：输出位置、文本影响预测、**交错无未来泄漏**、prepend 对照泄漏、文本投影梯度回传。

---

## 6. 实测参考

**路线 B（20 标的 / 200 iter / 1 epoch，CPU）**

| 阶段 | 文本覆盖率 | 说明 |
|------|-----------|------|
| 仅因子文本 | 3.2% | 龙虎榜/题材是事件驱动，天然稀疏 |
| + 新闻入库后 | **31.8%** | 抓取 20 只标的共 2031 条新闻，入库 1396 条 |

> 新闻把覆盖率提升约 10 倍 —— **新闻语料是前期融合有效性的关键**。

**路线 A CPU 冒烟**：tokenizer val loss ~0.17 / predictor ~3.04（20 标的，仅验证流程）。

---

## 7. 常见问题

**Q：文本覆盖率 0%？**
价格训练区间与文本（因子/新闻）日期不重叠。按 §3 把区间指到与文本重叠的时段。

**Q：想接入自有新闻数据？**
按 §2.2 的 `news` 表结构写入 `sequoia_v2.db` 即可，管线自动识别。

**Q：新闻抓不到历史（2024 年）的？**
免费财经接口通常只提供近期新闻。建议**定期运行 `fetch_news.py` 增量累积**，
或把训练窗口设到有文本的近期区间。

**Q：CPU 下单进程报 "Default process group has not been initialized"？**
某处 `dist.*` 调用未加 `if dist.is_initialized():` 守卫。已修复的都在
`train_tokenizer.py` / `train_predictor.py` 内，新增脚本请沿用同一模式。

**Q：checkpoint 被覆盖？**
冒烟与中等压测共用 `./outputs/models/...`；融合模型单独存在
`sequoia_predictor_fusion` 下。需要多份请用不同 `SEQUOIA_CPU_DATASET_PATH` 并备份。

**Q：`pytest` / `transformers` 未安装？**
两者都是可选的：`tests/test_kronos_fusion.py` 无 pytest 时会自行逐个执行；
`transformers` 缺失时融合会自动降级为 hash 嵌入。

---

## 8. 相关文档

- `finetune/CPU_FINETUNE_GUIDE.md` —— 路线 A 的 CPU 适配细节
- `finetune/EARLY_FUSION_DESIGN.md` —— 路线 B 的架构设计、因果性论证与已知限制
