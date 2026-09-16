# Kronos 扩展维度微调 · CPU 使用指南

本项目在预训练权重上**引入新维度**(在原始 6 维 open/high/low/close/vol/amt 基础上额外加入 `outstanding_share`,构成 7 维 feature),对 `KronosTokenizer` 与 `KronosPredictor` 做微调。本指南说明如何在**无 GPU 的 Mac / CPU 环境**上跑通这条「扩展维度微调」流程。

> 说明:CPU 仅为**流程验证与小规模压测**之用。真实全量训练(5000+ 标的 / 30 epoch)必须在 GPU 机器上用 `run_sequoia_finetune.sh` 完成。

---

## 1. 两种预制规模

| 脚本 | 规模 | 耗时 | 用途 |
|------|------|------|------|
| `bash run_sequoia_cpu.sh` | 20 标的 / 200 样本上限 / 1 epoch | ~30s | 纯流程冒烟验证(确认能端到端跑通) |
| `bash run_sequoia_cpu_medium.sh` | 120 标的 / 满样本 / 3 epoch | ~3h | 大数据量稳定性压测 |

两个脚本都会依次执行三步:
1. `sequoia_db_loader.py` —— 从 `data/sequoia_v2.db` 生成 pickle 数据集
2. `train_tokenizer.py` —— 微调扩展 tokenizer(复用原有权重 + 新维度)
3. `train_predictor.py` —— 微调 predictor(加载上一步的扩展 tokenizer)

---

## 2. 自定义规模(无需改代码)

`finetune/config_sequoia_cpu.py` 已支持以下环境变量覆盖,直接命令行传入即可:

| 环境变量 | 含义 | 默认值 |
|----------|------|--------|
| `SEQUOIA_CPU_DATASET_PATH` | 数据集输出/读取目录 | `./data/sequoia_cpu` |
| `SEQUOIA_CPU_MAX_SYMBOLS` | 选取的标的数量 | `20` |
| `SEQUOIA_CPU_EPOCHS` | epoch 数 | `1` |
| `SEQUOIA_CPU_TRAIN_ITER` | 单 epoch 训练样本上限(设很大 = 用满全部自然样本) | `200` |
| `SEQUOIA_CPU_VAL_ITER` | 单 epoch 验证样本上限 | `40` |

### 示例:跑 200 标的、5 epoch

```bash
export SEQUOIA=1
export SEQUOIA_CPU=1
export SEQUOIA_CPU_DATASET_PATH=./data/sequoia_cpu_200
export SEQUOIA_CPU_MAX_SYMBOLS=200
export SEQUOIA_CPU_EPOCHS=5
export SEQUOIA_CPU_TRAIN_ITER=1000000     # 用满自然样本,不封顶
export SEQUOIA_CPU_VAL_ITER=200000
export PYTHONPATH="$(pwd):$PYTHONPATH"

python -u finetune/sequoia_db_loader.py
python -u finetune/train_tokenizer.py
python -u finetune/train_predictor.py
```

> 想快速换规模,直接复用 `run_sequoia_cpu_medium.sh`,改一下里面的 `export` 即可。

---

## 3. 工作原理(为什么 CPU 不需要 torchrun)

- `SEQUOIA_CPU=1` 时,`setup_ddp()` 走 **CPU 单进程分支**:`device='cpu'`、`backend='gloo'`、`world_size=1`。
- 所有 `torch.distributed` 调用(`dist.barrier()` / `dist.all_reduce()` 等)均被 `if dist.is_initialized():` 守卫 —— CPU 单进程下 `is_initialized()` 为 `False`,直接跳过,不报错。
- 因此**无需 torchrun**,直接 `python -u finetune/train_*.py` 即可单机单进程运行。

---

## 4. 输出 / 监控 / 停止

- **数据集**:`sequoia_db_loader.py` 生成 `train_data.pkl` / `val_data.pkl` / `test_data.pkl` 到 `config.dataset_path`(各规模建议用不同目录隔离,见上方环境变量)。
- **权重**:
  - `./outputs/models/sequoia_tokenizer/checkpoints/best_model`
  - `./outputs/models/sequoia_predictor/checkpoints/best_model`
  - ⚠️ 冒烟与中等压测**共用**这两个路径,会互相覆盖。若需保留多份,用 `SEQUOIA_CPU_DATASET_PATH` 区分,并另行备份对应 checkpoint。
- **实时日志**:脚本已用 `python -u`(无缓冲),可直接 `tail -f /tmp/sequoia_medium.log` 观察进度。
- **中途停止**:`pkill -f run_sequoia_cpu_medium.sh`(或对应脚本名)。

---

## 5. GPU 真实全量训练

```bash
bash run_sequoia_finetune.sh    # 使用 config_sequoia.py:5000+ 标的 / 30 epoch,需 torchrun 多卡
```

CPU 适配代码已与 GPU 路径隔离(`SEQUOIA_CPU` 环境变量控制),改动不影响原有多卡训练流程。

---

## 6. 已修复的 CPU 兼容性问题(供参考)

原代码在 CPU 单进程下会崩溃,均已在 `train_tokenizer.py` / `train_predictor.py` 中修复:

1. `setup_ddp()` 默认 `nccl` + `cuda.set_device`,CPU 下改为 `gloo` + `device='cpu'`。
2. `__main__` 曾强制要求 `WORLD_SIZE`(即必须 torchrun);现 `SEQUOIA_CPU=1` 时 `world_size=1`,可直接运行。
3. 多处 `dist.barrier()` / `dist.all_reduce()` 未守卫,已统一包在 `if dist.is_initialized():` 内。

若后续新增训练脚本,务必沿用相同守卫模式,否则单进程 CPU 会报 "Default process group has not been initialized"。
