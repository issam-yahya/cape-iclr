"""
Long-term-forecasting datasets.

Self-contained subset of the Time-Series-Library data loaders — only the three
classes CAPE needs (ETT-hour, ETT-minute, Custom). Splits, scaling and time
encodings are byte-for-byte identical to the reference implementation, so the
numbers reported in the paper are directly comparable to published baselines.

Additions over the reference loaders:
  * `percent` — keep only the first `percent`% of the training split
    (used for the 10% few-shot experiments in Table 5). Validation and test
    splits are never subsampled.
  * CSVs are read from disk; if a file is missing we fall back to the
    HuggingFace mirror of the benchmark (optional dependency).
"""

import os
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from data_provider.timefeatures import time_features

warnings.filterwarnings('ignore')

HUGGINGFACE_REPO = "thuml/Time-Series-Library"


def _read_csv(root_path, data_path):
    """Read the benchmark CSV, falling back to the HuggingFace mirror."""
    local_fp = os.path.join(root_path, data_path)
    if os.path.exists(local_fp):
        return pd.read_csv(local_fp)

    cfg_name = os.path.splitext(os.path.basename(data_path))[0]
    try:
        from datasets import load_dataset
    except ImportError:
        raise FileNotFoundError(
            f"Could not find {local_fp}.\n"
            f"Either place the CSV there (see nips_code/README.md → Data), or\n"
            f"install the optional downloader:  pip install datasets"
        )
    ds = load_dataset(HUGGINGFACE_REPO, name=cfg_name)
    split_name = "train" if "train" in ds else list(ds.keys())[0]
    return ds[split_name].to_pandas()


def _build_stamp(df_raw, border1, border2, timeenc, freq, minute_feature=False):
    """Time-feature encoding of the date column over [border1, border2)."""
    df_stamp = df_raw[['date']][border1:border2]
    df_stamp['date'] = pd.to_datetime(df_stamp.date)
    if timeenc == 0:
        df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
        df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
        df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
        df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
        if minute_feature:
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
        return df_stamp.drop(['date'], axis=1).values
    data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=freq)
    return data_stamp.transpose(1, 0)


class _ForecastDataset(Dataset):
    """Shared windowing logic: X = [t, t+L), Y = [t+L-label_len, t+L+H)."""

    minute_feature = False

    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',
                 seasonal_patterns=None):
        self.args = args
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len, self.label_len, self.pred_len = size

        assert flag in ['train', 'test', 'val']
        self.set_type = {'train': 0, 'val': 1, 'test': 2}[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        # `percent` only ever shrinks the training split
        self.percent = getattr(args, 'percent', 100)

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def _borders(self, df_raw):
        """Return (border1s, border2s) — the split boundaries."""
        raise NotImplementedError

    def _reorder(self, df_raw):
        return df_raw

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = self._reorder(_read_csv(self.root_path, self.data_path))

        border1s, border2s = self._borders(df_raw)
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # Few-shot: keep the first `percent`% of the training window only.
        if self.set_type == 0 and self.percent < 100:
            border2 = border1 + (border2 - border1) * self.percent // 100

        if self.features in ('M', 'MS'):
            df_data = df_raw[df_raw.columns[1:]]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
        else:
            raise ValueError(f"unknown features mode: {self.features}")

        if self.scale:
            # Scaler is always fit on the *full* training split, even in
            # few-shot mode, matching the standard benchmark protocol.
            self.scaler.fit(df_data[border1s[0]:border2s[0]].values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = _build_stamp(df_raw, border1, border2,
                                       self.timeenc, self.freq,
                                       minute_feature=self.minute_feature)

        n_windows = len(self.data_x) - self.seq_len - self.pred_len + 1
        if n_windows <= 0:
            raise ValueError(
                f"{type(self).__name__}({self.data_path}, "
                f"split={['train', 'val', 'test'][self.set_type]}, "
                f"percent={self.percent}) yields no usable windows: the split has "
                f"{len(self.data_x)} timesteps but seq_len + pred_len = "
                f"{self.seq_len} + {self.pred_len} = {self.seq_len + self.pred_len} "
                f"are needed per window.\n"
                f"Reduce --seq_len or --pred_len, or raise --percent."
            )

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return max(0, len(self.data_x) - self.seq_len - self.pred_len + 1)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_hour(_ForecastDataset):
    """ETTh1 / ETTh2 — 12/4/4 months train/val/test."""

    def _borders(self, df_raw):
        border1s = [0,
                    12 * 30 * 24 - self.seq_len,
                    12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24,
                    12 * 30 * 24 + 4 * 30 * 24,
                    12 * 30 * 24 + 8 * 30 * 24]
        return border1s, border2s


class Dataset_ETT_minute(_ForecastDataset):
    """ETTm1 / ETTm2 — same 12/4/4 months at 15-minute resolution."""

    minute_feature = True

    def _borders(self, df_raw):
        border1s = [0,
                    12 * 30 * 24 * 4 - self.seq_len,
                    12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4,
                    12 * 30 * 24 * 4 + 4 * 30 * 24 * 4,
                    12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        return border1s, border2s


class Dataset_Custom(_ForecastDataset):
    """Weather / ECL / Traffic / Exchange — 70/10/20 split."""

    def _reorder(self, df_raw):
        # df_raw.columns: ['date', ...(other features), target feature]
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        return df_raw[['date'] + cols + [self.target]]

    def _borders(self, df_raw):
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0,
                    num_train - self.seq_len,
                    len(df_raw) - num_test - self.seq_len]
        border2s = [num_train,
                    num_train + num_vali,
                    len(df_raw)]
        return border1s, border2s
