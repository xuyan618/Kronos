"""
train_predictor_fusion.py —— 前期融合（Early Fusion）predictor 微调。

与 train_predictor.py 的区别：
    - 数据集换成 QlibFusionDataset，额外产出逐日对齐的文本嵌入；
    - 模型换成 KronosFusion（在预训练 Kronos 上加文本 token 通道）；
    - 前向时把文本嵌入作为额外 token 拼入，让 Transformer 做跨模态联合注意力。

CPU 单进程：SEQUOIA_CPU=1 时可直接 `python` 运行（无需 torchrun）；
GPU：需用 torchrun 启动。
"""

import json
import os
import pickle
import sys
import time
from time import gmtime, strftime

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

try:
    import comet_ml
except ImportError:
    comet_ml = None

sys.path.append('../')
from config import Config  # noqa: E402
from dataset_fusion import QlibFusionDataset  # noqa: E402
from model.kronos_fusion import build_fusion_from_pretrained  # noqa: E402
from utils.training_utils import (  # noqa: E402
    setup_ddp, cleanup_ddp, set_seed, get_model_size, format_time,
)


# 可选确定性（OOS 可复现）：固定线程数+种子，消除 CPU 多线程下弱信号预测的抖动。
if os.environ.get("SEQUOIA_DETERMINISTIC") == "1":
    _seed = int(os.environ.get("SEED", "0"))
    torch.manual_seed(_seed)
    try:
        import numpy as _np
        _np.random.seed(_seed)
    except Exception:
        pass
    torch.set_num_threads(1)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def get_text_dim(config):
    """从缓存的嵌入文件读取 text_dim，保证与模型 text_proj 输入维度一致。"""
    path = os.path.join(
        config['dataset_path'], config.get('text_embeddings_file', 'text_embeddings.pkl')
    )
    if os.path.exists(path):
        try:
            with open(path, 'rb') as f:
                return int(pickle.load(f).get('dim'))
        except Exception as e:
            print(f"[fusion] 读取嵌入维度失败({e})，回退到 text_hash_dim")
    return int(config.get('text_hash_dim', 128))


def create_dataloaders(config, rank, world_size):
    print(f"[Rank {rank}] Creating distributed dataloaders (fusion)...")
    train_dataset = QlibFusionDataset('train')
    valid_dataset = QlibFusionDataset('val')
    print(f"[Rank {rank}] Train dataset size: {len(train_dataset)}, "
          f"Validation dataset size: {len(valid_dataset)}")

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], sampler=train_sampler,
        num_workers=config.get('num_workers', 2),
        pin_memory=torch.cuda.is_available(), drop_last=True,
    )
    val_loader = DataLoader(
        valid_dataset, batch_size=config['batch_size'], sampler=val_sampler,
        num_workers=config.get('num_workers', 2),
        pin_memory=torch.cuda.is_available(), drop_last=False,
    )
    return train_loader, val_loader, train_dataset, valid_dataset


def train_model(model, tokenizer, device, config, save_dir, logger, rank, world_size):
    start_time = time.time()
    model_ref = model.module if isinstance(model, DDP) else model
    if rank == 0:
        print(f"Effective BATCHSIZE per GPU: {config['batch_size']}, "
              f"Total: {config['batch_size'] * world_size}")

    train_loader, val_loader, train_dataset, valid_dataset = create_dataloaders(config, rank, world_size)

    optimizer = torch.optim.AdamW(
        [pp for pp in model.parameters() if pp.requires_grad],
        lr=config['predictor_learning_rate'],
        betas=(config['adam_beta1'], config['adam_beta2']),
        weight_decay=float(os.environ.get(
            "FUSION_WEIGHT_DECAY", config.get('adam_weight_decay', 0.01))),
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config['predictor_learning_rate'],
        steps_per_epoch=len(train_loader), epochs=config['epochs'],
        pct_start=0.03, div_factor=10,
    )

    best_val_loss = float('inf')
    dt_result = {}
    batch_idx_global = 0

    for epoch_idx in range(config['epochs']):
        epoch_start_time = time.time()
        model.train()
        train_loader.sampler.set_epoch(epoch_idx)
        train_dataset.set_epoch_seed(epoch_idx * 10000 + rank)
        valid_dataset.set_epoch_seed(0)

        for i, (batch_x, batch_x_stamp, batch_text) in enumerate(train_loader):
            batch_x = batch_x.to(device, non_blocking=True)
            batch_x_stamp = batch_x_stamp.to(device, non_blocking=True)
            batch_text = batch_text.to(device, non_blocking=True)

            with torch.no_grad():
                token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)

            token_in = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
            token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]
            # 文本与输入 token 逐日对齐（去掉最后一天，避免用当天文本预测当天）
            text_in = batch_text[:, :-1, :]

            logits = model(token_in[0], token_in[1], batch_x_stamp[:, :-1, :], text_emb=text_in)
            loss, s1_loss, s2_loss = model_ref.head.compute_loss(
                logits[0], logits[1], token_out[0], token_out[1]
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()
            scheduler.step()

            if rank == 0 and (batch_idx_global + 1) % config['log_interval'] == 0:
                lr = optimizer.param_groups[0]['lr']
                print(
                    f"[Rank {rank}, Epoch {epoch_idx + 1}/{config['epochs']}, "
                    f"Step {i + 1}/{len(train_loader)}] LR {lr:.6f}, Loss: {loss.item():.4f}"
                )
            if rank == 0 and logger:
                logger.log_metric('train_fusion_loss_batch', loss.item(), step=batch_idx_global)

            batch_idx_global += 1
            if batch_idx_global >= config['n_train_iter']:
                break

        # ---- 验证 ----
        model.eval()
        tot_val_loss_sum_rank = 0.0
        val_batches_processed_rank = 0
        with torch.no_grad():
            for batch_x, batch_x_stamp, batch_text in val_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_x_stamp = batch_x_stamp.to(device, non_blocking=True)
                batch_text = batch_text.to(device, non_blocking=True)

                token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)
                token_in = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
                token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]

                logits = model(
                    token_in[0], token_in[1], batch_x_stamp[:, :-1, :],
                    text_emb=batch_text[:, :-1, :],
                )
                val_loss, _, _ = model_ref.head.compute_loss(
                    logits[0], logits[1], token_out[0], token_out[1]
                )
                tot_val_loss_sum_rank += val_loss.item()
                val_batches_processed_rank += 1
                if val_batches_processed_rank >= config['n_val_iter']:
                    break

        val_loss_sum_tensor = torch.tensor(tot_val_loss_sum_rank, device=device)
        val_batches_tensor = torch.tensor(val_batches_processed_rank, device=device)
        if dist.is_initialized():
            dist.all_reduce(val_loss_sum_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_batches_tensor, op=dist.ReduceOp.SUM)

        avg_val_loss = (
            val_loss_sum_tensor.item() / val_batches_tensor.item()
            if val_batches_tensor.item() > 0 else 0
        )

        if rank == 0:
            print(f"\n--- Epoch {epoch_idx + 1}/{config['epochs']} Summary ---")
            print(f"Validation Loss: {avg_val_loss:.4f}")
            print(f"Time This Epoch: {format_time(time.time() - epoch_start_time)}")
            print(f"Total Time Elapsed: {format_time(time.time() - start_time)}\n")
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_path = f"{save_dir}/checkpoints/best_model"
                model_ref.save_pretrained(save_path)
                print(f"Best model saved to {save_path} (Val Loss: {best_val_loss:.4f})")

        if dist.is_initialized():
            dist.barrier()

        if batch_idx_global >= config['n_train_iter']:
            break

    dt_result['best_val_loss'] = best_val_loss
    return dt_result


def main(config: dict):
    rank, world_size, local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
    set_seed(config['seed'], rank)

    save_dir = os.path.join(config['save_path'], config['predictor_save_folder_name'])

    comet_logger, master_summary = None, {}
    if rank == 0:
        os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)
        master_summary = {
            'start_time': strftime("%Y-%m-%dT%H-%M-%S", gmtime()),
            'save_directory': save_dir,
            'world_size': world_size,
            'fusion_mode': config.get('fusion_mode', 'interleave'),
            'text_dim': get_text_dim(config),
        }
        if config['use_comet'] and comet_ml is not None:
            comet_logger = comet_ml.Experiment(
                api_key=config['comet_config']['api_key'],
                project_name=config['comet_config']['project_name'],
                workspace=config['comet_config']['workspace'],
            )
            comet_logger.add_tag(config['comet_tag'])
            comet_logger.set_name(config['comet_name'])
            comet_logger.log_parameters(config)

    if dist.is_initialized():
        dist.barrier()

    text_dim = get_text_dim(config)
    if rank == 0:
        print(f"[fusion] text_dim={text_dim}, fusion_mode={config.get('fusion_mode')}")

    # Tokenizer（扩展维度版本，冻结）
    from model_factory import build_tokenizer
    tokenizer = build_tokenizer(config, finetuned=True)
    tokenizer.eval().to(device)

    # 融合模型：backbone 复用预训练权重，文本通道随机初始化
    model = build_fusion_from_pretrained(
        config['pretrained_predictor_path'],
        text_dim=text_dim,
        fusion_mode=config.get('fusion_mode', 'interleave'),
        use_modality_emb=config.get('use_modality_emb', True),
        device=device,
    )

    # 可选：冻结 backbone 以缓解过拟合（仅训练文本通道 + head + dep_layer + norm）
    if os.environ.get("FREEZE_BACKBONE") == "1":
        frozen = 0
        for _n, _p in model.backbone.named_parameters():
            if _n.split(".")[0] in ("transformer", "embedding", "time_emb"):
                _p.requires_grad = False
                frozen += 1
        print(f"[fusion] 冻结 backbone 参数 {frozen} 个（仅训练文本通道+head+dep_layer+norm）")

    use_ddp = torch.cuda.is_available()
    if use_ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    if rank == 0:
        print(f"Fusion Predictor Model Size: {get_model_size(model.module if use_ddp else model)}")

    dt_result = train_model(
        model, tokenizer, device, config, save_dir, comet_logger, rank, world_size
    )

    if rank == 0:
        master_summary['final_result'] = dt_result
        with open(os.path.join(save_dir, 'summary.json'), 'w') as f:
            json.dump(master_summary, f, indent=4)
        print('Training finished. Summary file saved.')
        if comet_logger:
            comet_logger.end()

    cleanup_ddp()


if __name__ == '__main__':
    # GPU 训练需用 torchrun；CPU 单进程可直接 python 运行
    if torch.cuda.is_available() and "WORLD_SIZE" not in os.environ:
        raise RuntimeError("GPU training must be launched with `torchrun`.")

    if os.environ.get("SEQUOIA_FUSION") == "1":
        from config_sequoia_fusion import get_config as _get_config
        config_instance = _get_config()
    elif os.environ.get("SEQUOIA_CPU") == "1":
        from config_sequoia_cpu import get_config as _get_config
        config_instance = _get_config()
    elif os.environ.get("SEQUOIA") == "1":
        from config_sequoia import get_config as _get_config
        config_instance = _get_config()
    else:
        config_instance = Config()
    main(config_instance.__dict__)
