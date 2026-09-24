# CAPE: Closed-form Analytic Period-folded Estimation

Reproducibility code for the ICLR submission.

This folder is self-contained. Copy it anywhere, add the data, and run.

## What is CAPE?

Zero-parameter time-series forecasting model. No learned weights; solved analytically via ridge regression.

Algorithm (one dataset, one pred_len):

1. Detect period `p` (FFT + autocorrelation).
2. Instance-normalise each window.
3. Moving-average smoother, window = p.
4. Period-fold: reshape `[L, C] -> [p, S]`, `S = L / p`.
5. Accumulate normal equations `A^T A`, `A^T B`.
6. Solve: `W = (A^T A + λ I)^{-1} A^T B`.
7. Forecast: `Ŷ = X @ W`, reshaped and truncated to `pred_len`.

No gradients. No GPU. Fits in seconds.

## Folder layout

```
nips_code/
  cape_code/
    cape.py              CAPE model
    run_forecasting.py   Table 2, Table 5
    run_zeroshot.py      Table 6
  cape_legacy/           original scripts (reference only)
  data_provider/
    data_factory.py      dataset registry
    data_loader.py       ETT-hour / ETT-minute / Custom loaders
    timefeatures.py      calendar feature encoding
  dataset/               place benchmark CSVs here
  results/               output .txt files
  requirements.txt
  README.md
```

`data_provider/` is a trimmed copy from [Time-Series-Library](https://github.com/thuml/Time-Series-Library). Only long-term-forecasting loaders are kept.

`cape_legacy/` is the original, unmodified scripts, kept for provenance. Run `cape_code/`; `cape_legacy/` is reference only.

## Installation

```bash
pip install -r requirements.txt
```

torch, numpy, pandas, scikit-learn.

## Data

`dataset/` ships empty. Populate with the standard long-term forecasting benchmarks:

```
nips_code/dataset/
  ETT-small/
    ETTh1.csv  ETTh2.csv  ETTm1.csv  ETTm2.csv
  weather/
    weather.csv
  electricity/
    electricity.csv
  traffic/
    traffic.csv
  exchange_rate/
    exchange_rate.csv
```

Download from the Time-Series-Library Google Drive:

> https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy

Or point at an existing copy:

```bash
python cape_code/run_forecasting.py --table_full --data_root /path/to/parent-of-dataset
```

## Running

```bash
cd nips_code
```

### Table 2

```bash
python cape_code/run_forecasting.py --table_full --seq_len 720 --ridge_alpha 1.0
```

### Table 5

```bash
python cape_code/run_forecasting.py --table_fewshot --seq_len 96 --ridge_alpha 1.0
```

### Table 6

```bash
python cape_code/run_zeroshot.py --table6 --seq_len 720 --ridge_alpha 1.0
```

## Single-dataset / custom runs

```bash
python cape_code/run_forecasting.py \
    --data ETTh1 \
    --root_path dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --enc_in 7 --freq h \
    --seq_len 720 --ridge_alpha 1.0
```

```bash
python cape_code/run_forecasting.py --data ETTh1 --pred_len 96 --sweep_alpha
```

```bash
python cape_code/run_zeroshot.py \
    --src_data ETTh1 --src_path dataset/ETT-small/ --src_file ETTh1.csv \
    --tgt_data ETTh2 --tgt_path dataset/ETT-small/ --tgt_file ETTh2.csv \
    --enc_in 7 --pred_len 96
```

## Key hyperparameters

- `seq_len` (default 720): lookback window length
- `pred_len` (default 96): forecast horizon
- `ridge_alpha` (default 1.0): ridge regularisation λ
- `period_len` (default -1): period p; -1 = auto-detect
- `max_period` (default -1): cap auto-detection
- `min_period` (default 2): floor for period auto-detection
- `percent` (default 100): % of training data
- `data_root` (default `nips_code/`): parent directory of `dataset/`
- `results_dir` (default `nips_code/results/`): output directory
- `verbose` (default off): print diagnostics

```bash
python cape_code/run_forecasting.py --data ETTh1 --pred_len 96 --verbose
```

## Using the CAPE class directly

```python
import sys
sys.path.insert(0, 'nips_code/cape_code')

import torch
from cape import CAPE

class Config:
    seq_len     = 720
    pred_len    = 96
    enc_in      = 7
    ridge_alpha = 1.0
    period_len  = -1
    max_period  = -1

model = CAPE(Config())
model.fit(train_loader, device='cpu')

with torch.no_grad():
    pred = model(batch_x)
```
