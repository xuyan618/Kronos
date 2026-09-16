"""
dataset_fusion.py —— 前期融合用数据集。

在 QlibDataset 基础上，除 (x, x_stamp) 外，额外返回与窗口**逐日对齐**的文本嵌入
text: [window, text_dim]，供 KronosFusion 当作额外 token 拼入输入序列。

对齐方式
--------
父类为省内存会把 datetime 列裁掉，因此这里重新读取一次原始 pickle 拿到日期索引，
再按 (symbol, date) 查表得到每天的嵌入；查不到的日期填全零（表示「当天无文本」）。
"""

import os
import pickle

import numpy as np
import torch

from dataset import QlibDataset


class QlibFusionDataset(QlibDataset):
    """返回 (x, x_stamp, text) 的融合数据集。"""

    def __init__(self, data_type='train', text_emb_path=None):
        super().__init__(data_type)

        # 1) 重新读取原始 pickle 以拿到日期索引（父类已裁掉 datetime 列）
        with open(self.data_path, 'rb') as f:
            raw = pickle.load(f)

        # 2) 载入文本嵌入包
        path = text_emb_path or os.path.join(
            self.config.dataset_path,
            getattr(self.config, 'text_embeddings_file', 'text_embeddings.pkl'),
        )
        pack = None
        if os.path.exists(path):
            with open(path, 'rb') as f:
                pack = pickle.load(f)
            self.text_dim = int(pack.get('dim', 128))
        else:
            self.text_dim = int(getattr(self.config, 'text_hash_dim', 128))
            print(f"[fusion-dataset] WARNING: 未找到文本嵌入 {path}，将全部使用全零文本向量")

        data = (pack or {}).get('data', {}) if pack else {}

        # 3) 构建 symbol -> [n_rows, text_dim] 的按行对齐矩阵
        self.text_by_symbol = {}
        filled_rows = 0
        total_rows = 0
        for symbol in self.symbols:
            df_raw = raw.get(symbol)
            if df_raw is None:
                continue
            n_rows = len(df_raw)
            mat = np.zeros((n_rows, self.text_dim), dtype=np.float32)

            sym_pack = data.get(symbol)
            if sym_pack is not None:
                dates = sym_pack['dates']
                embs = sym_pack['emb']
                date_to_row = {d: i for i, d in enumerate(dates)}
                try:
                    idx_keys = df_raw.index.strftime('%Y-%m-%d')
                except Exception:
                    idx_keys = [str(d)[:10] for d in df_raw.index]
                for row_i, dkey in enumerate(idx_keys):
                    j = date_to_row.get(dkey)
                    if j is not None:
                        mat[row_i] = embs[j]
                        filled_rows += 1
            total_rows += n_rows
            self.text_by_symbol[symbol] = mat

        if total_rows:
            print(f"[fusion-dataset] 文本覆盖率: {filled_rows}/{total_rows} 行 "
                  f"({filled_rows / total_rows:.1%})，text_dim={self.text_dim}")

    def __getitem__(self, idx):
        # 与父类一致地随机取一个 (symbol, start_idx) 窗口
        random_idx = self.py_rng.randint(0, len(self.indices) - 1)
        symbol, start_idx = self.indices[random_idx]

        df = self.data[symbol]
        end_idx = start_idx + self.window
        win_df = df.iloc[start_idx:end_idx]

        x = win_df[self.feature_list].values.astype(np.float32)
        x_stamp = win_df[self.time_feature_list].values.astype(np.float32)

        # 归一化：仅用 lookback 窗口的均值/标准差，防止未来泄漏
        past_len = self.config.lookback_window
        past_x = x[:past_len]
        x_mean = np.mean(past_x, axis=0)
        x_std = np.std(past_x, axis=0)
        x = (x - x_mean) / (x_std + 1e-5)
        x = np.clip(x, -self.config.clip, self.config.clip)

        # 与窗口逐日对齐的文本嵌入
        mat = self.text_by_symbol.get(symbol)
        if mat is None:
            text = np.zeros((self.window, self.text_dim), dtype=np.float32)
        else:
            seg = mat[start_idx:end_idx]
            if seg.shape[0] < self.window:
                pad = np.zeros((self.window - seg.shape[0], self.text_dim), dtype=np.float32)
                seg = np.concatenate([seg, pad], axis=0)
            text = seg.astype(np.float32)

        return torch.from_numpy(x), torch.from_numpy(x_stamp), torch.from_numpy(text)
