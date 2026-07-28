# CAPE: Closed-form Analytic Period-folded Estimation

Reproducibility code for the NeurIPS submission.

This folder is **self-contained** — it does not depend on the surrounding
repository. Copy it anywhere, add the data, and run.

## What is CAPE?

CAPE is a **zero-parameter** time-series forecasting model.
It has no learned weights — everything is determined analytically via ridge regression.

**Algorithm** (for one dataset, one pred_len):

1. Detect the dominant period `p` from the training data (FFT + autocorrelation).
2. Instance-normalise each window (subtract per-sequence mean).
3. Apply a fixed moving-average smoother (window = p).
4. **Period-fold** the lookback window: reshape `[L, C] → [p, S]` where `S = L / p`.
5. Accumulate the normal equations `A^T A` and `A^T B` over all training windows.
6. Solve analytically: `W = (A^T A + λ I)^{-1} A^T B`  — one matrix solve, no iterations.
7. Forecast: `Ŷ = X @ W`, reshaped and truncated to `pred_len`.

**No gradients. No GPU required. Fits in seconds.**

---

## Folder layout

```
nips_code/
  cape_code/
    cape.py              — CAPE model (self-contained, depends only on PyTorch)
    run_forecasting.py   — Table 2 (full data) and Table 5 (10% few-shot)
    run_zeroshot.py      — Table 6 (cross-dataset zero-shot)
  cape_legacy/           — the original scripts, untouched (see below)
  data_provider/
    data_factory.py      — dataset registry + DataLoader construction
    data_loader.py       — ETT-hour / ETT-minute / Custom loaders
    timefeatures.py      — calendar feature encoding
  dataset/               — EMPTY: place the benchmark CSVs here (see below)
  results/               — output .txt files are written here
  requirements.txt
  README.md              — this file
```

`data_provider/` is a trimmed copy of the corresponding module from the
[Time-Series-Library](https://github.com/thuml/Time-Series-Library). Splits,
scaling and time encodings are unchanged, so CAPE's numbers are directly
comparable to published baselines. Only the long-term-forecasting loaders are
kept (the anomaly-detection / classification / M4 branches and their `sktime`
dependency are not needed here), plus support for the `--percent` flag used by
the few-shot experiments.

`cape_legacy/` is the original, unmodified version of the three scripts, kept
for provenance. `cape.py` there is byte-identical to `cape_code/cape.py` — the
model was never touched. The two `run_*.py` files differ only in path handling:
the legacy versions resolve `dataset/` and `results/` relative to the *current
working directory*, so they must be run from `nips_code/`:

```bash
cd nips_code
python cape_legacy/run_forecasting.py --data ETTh1 --pred_len 96
```

Both versions produce identical numbers (verified: ETTh1 `H=96` → MSE 0.386166
from each). **Run `cape_code/` — `cape_legacy/` is a reference copy only.**

---

## Installation

```bash
pip install -r requirements.txt
```

That is `torch`, `numpy`, `pandas`, `scikit-learn` — nothing else is required.

---

## Data

The `dataset/` folder ships **empty**. Populate it with the standard long-term
forecasting benchmarks:

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

**Download.** These are the benchmarks used across the time-series community.
Get them from the Time-Series-Library Google Drive:

> https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy

Unzip and place the CSVs exactly as shown. The folder names (`ETT-small`,
`weather`, …) must match — the scripts reference them by those exact paths.

Verify before running:

```bash
ls dataset/ETT-small/    # ETTh1.csv ETTh2.csv ETTm1.csv ETTm2.csv
ls dataset/weather/      # weather.csv
ls dataset/electricity/  # electricity.csv
```

*Alternative:* `pip install datasets` and the loaders will pull any missing CSV
from the [HuggingFace mirror](https://huggingface.co/datasets/thuml/Time-Series-Library)
automatically, leaving `dataset/` untouched.

**Data already elsewhere?** Point the scripts at it instead of copying:

```bash
python cape_code/run_forecasting.py --table_full --data_root /path/to/parent-of-dataset
```

---

## Running

All commands below are written from the `nips_code/` directory, but the scripts
resolve their paths from their own location — they work from **any** working
directory:

```bash
cd nips_code
python cape_code/run_forecasting.py --table_full
```

Add `--no_cuda` to force CPU (CAPE is fast enough on CPU that this is the
recommended default).

### Table 2 — Full-data forecasting

```bash
python cape_code/run_forecasting.py --table_full --seq_len 720 --ridge_alpha 1.0
```

Results are written to `results/table_full_a1.0.txt`.

Avg MSE over pred_lens 96/192/336/720. "Paper" is the value reported in the
submission; "reproduced" is what this code produced on CPU (see
[Reproduction status](#reproduction-status)).

| Dataset  | Paper | Reproduced | Δ      | SOTA (TimeMixer++) |
|----------|-------|------------|--------|--------------------|
| ETTh1    | 0.390 | 0.415      | +0.025 | 0.357              |
| ETTh2    | 0.357 | 0.333      | −0.024 | 0.331              |
| ETTm1    | 0.376 | 0.356      | −0.020 | 0.373              |
| ETTm2    | 0.272 | 0.249      | −0.023 | 0.253              |
| ETT(Avg) | 0.349 | 0.338      | −0.011 | 0.329              |
| Weather  | 0.252 | not re-run |        | 0.225              |
| ECL      | 0.213 | not re-run |        | 0.161              |
| Exchange | 0.088 | not re-run |        | —                  |

Every ETT dataset is off by roughly ±0.02 — three of them *better* than
reported, ETTh1 worse. On the reproduced numbers CAPE beats TimeMixer++ on
ETTm1 (0.356 vs 0.373) and ETTm2 (0.249 vs 0.253), and ties on ETTh2.

### Table 5 — 10% few-shot forecasting

```bash
python cape_code/run_forecasting.py --table_fewshot --seq_len 720 --ridge_alpha 1.0
```

Results are written to `results/table_fewshot_a1.0.txt`.

> **⚠ This experiment does not run at `--seq_len 720`.** ETT's training split is
> 8640 timesteps, so 10% of it is 864. A window needs `seq_len + pred_len`
> timesteps, i.e. 816 / 912 / 1056 / 1440 for the four horizons — only `H=96`
> fits. The command above therefore raises a "no usable windows" error on ETTh1
> at `H=192`. Use a shorter lookback for this experiment:
>
> ```bash
> python cape_code/run_forecasting.py --table_fewshot --seq_len 96 --ridge_alpha 1.0
> ```

Avg MSE over pred_lens 96/192/336/720, at `--seq_len 96`:

| Dataset  | Paper | Reproduced | TimeMixer++ (SOTA) |
|----------|-------|------------|--------------------|
| ETTh1    | —     | 0.853      | —                  |
| ETTh2    | —     | 0.943      | —                  |
| ETTm1    | —     | 0.520      | —                  |
| ETTm2    | —     | 0.291      | —                  |
| ETT(Avg) | 0.382 | **0.652**  | 0.396              |
| Weather  | 0.275 | 0.279      | 0.241              |
| ECL      | 0.228 | not re-run | 0.168              |

Weather reproduces (0.279 vs 0.275) — it has enough data that 10% is still a
large training set. The ETT numbers do **not** reproduce, and the paper's claim
that CAPE beats TimeMixer++ on ETT(Avg) in the 10% setting is not supported by
this code. See [Reproduction status](#reproduction-status).

### Table 6 — Cross-dataset zero-shot

Fit on one ETT dataset, evaluate on another **without any fine-tuning**.
This is possible because `W` lives in segment space, which is
channel-count-agnostic.

```bash
python cape_code/run_zeroshot.py --table6 --seq_len 720 --ridge_alpha 1.0
```

Results are written to `results/table6_zeroshot_a1.0.txt`.

Avg MSE over pred_lens 96/192/336/720. Reproduced on CPU with the default
`--min_period 2`; the last column shows what an earlier hardcoded floor of
`p >= 8` gives instead (see [Period floor](#period-floor)).

| Source → Target | Paper | Reproduced | with `min_period 8` |
|-----------------|-------|------------|---------------------|
| ETTh1 → ETTh2   | 0.349 | **0.349**  | 0.353               |
| ETTh1 → ETTm2   | 0.295 | **0.296**  | 0.297               |
| ETTh2 → ETTh1   | 0.417 | **0.417**  | 0.420               |
| ETTm1 → ETTh2   | 0.355 | **0.355**  | 0.411               |
| ETTm1 → ETTm2   | 0.256 | **0.256**  | 0.259               |
| ETTm2 → ETTm1   | 0.404 | **0.404**  | 0.409               |

All six pairs reproduce to three decimals.

---

## Single-dataset / custom runs

### One dataset, all pred_lens

```bash
python cape_code/run_forecasting.py \
    --data ETTh1 \
    --root_path dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --enc_in 7 --freq h \
    --seq_len 720 --ridge_alpha 1.0
```

### Sweep ridge_alpha

```bash
python cape_code/run_forecasting.py --data ETTh1 --pred_len 96 --sweep_alpha
```

### Single cross-domain pair

```bash
python cape_code/run_zeroshot.py \
    --src_data ETTh1 --src_path dataset/ETT-small/ --src_file ETTh1.csv \
    --tgt_data ETTh2 --tgt_path dataset/ETT-small/ --tgt_file ETTh2.csv \
    --enc_in 7 --pred_len 96
```

---

## Key hyperparameters

| Parameter     | Default | Description                                      |
|---------------|---------|--------------------------------------------------|
| `seq_len`     | 720     | Lookback window length                           |
| `pred_len`    | 96      | Forecast horizon                                 |
| `ridge_alpha` | 1.0     | Ridge regularisation λ                           |
| `period_len`  | -1      | Period p; -1 = auto-detect via FFT + ACF         |
| `max_period`  | -1      | Cap auto-detection; -1 = no cap                  |
| `min_period`  | 2       | Floor for period auto-detection                  |
| `percent`     | 100     | % of training data (10 = few-shot)               |
| `data_root`   | `nips_code/` | Parent directory of `dataset/`               |
| `results_dir` | `nips_code/results/` | Where result files are written      |
| `verbose`     | off     | Print period detection + fit diagnostics         |

By default the scripts print only the result lines. Pass `--verbose` to also see
the detected period, the candidate scores, the split sizes and `‖W‖`:

```bash
python cape_code/run_forecasting.py --data ETTh1 --pred_len 96 --verbose
```

---

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
    period_len  = -1    # auto
    max_period  = -1

model = CAPE(Config())

# fit once on the training DataLoader
model.fit(train_loader, device='cpu')

# inference
with torch.no_grad():
    pred = model(batch_x)   # [B, pred_len, C]
```

---
