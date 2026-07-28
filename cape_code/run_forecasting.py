"""
CAPE — Long-Term Forecasting Evaluation Script

Reproduces Table 2 (full training data) and Table 5 (10% few-shot) from the paper.

Quick start
-----------
# Full-data sweep over all 7 benchmark datasets (ETTh1/h2, ETTm1/m2, Weather, ECL, Exchange):
    python run_forecasting.py --table_full

# 10% few-shot sweep (Table 5):
    python run_forecasting.py --table_fewshot

# Single dataset, single pred_len:
    python run_forecasting.py --data ETTh1 --pred_len 96

# Sweep ridge_alpha values for a dataset:
    python run_forecasting.py --data ETTh1 --sweep_alpha

Dataset layout expected under --data_root  (default: ./dataset/)
    dataset/ETT-small/    ETTh1.csv  ETTh2.csv  ETTm1.csv  ETTm2.csv
    dataset/weather/      weather.csv
    dataset/electricity/  electricity.csv
    dataset/exchange_rate/ exchange_rate.csv
"""

import argparse
import copy
import os
import sys
import time

import numpy as np
import torch

# nips_code/ holds data_provider/ and dataset/; cape_code/ holds cape.py.
# Anchoring on __file__ means the script runs from any working directory.
PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PKG_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_provider.data_factory import data_provider
from cape import CAPE

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

PRED_LENS = [96, 192, 336, 720]

# (data_key, rel_root, filename, enc_in, freq, display_name)
ALL_DATASETS = [
    ('ETTh1',  'dataset/ETT-small/',   'ETTh1.csv',        7,   'h', 'ETTh1'),
    ('ETTh2',  'dataset/ETT-small/',   'ETTh2.csv',        7,   'h', 'ETTh2'),
    ('ETTm1',  'dataset/ETT-small/',   'ETTm1.csv',        7,   'm', 'ETTm1'),
    ('ETTm2',  'dataset/ETT-small/',   'ETTm2.csv',        7,   'm', 'ETTm2'),
    ('custom', 'dataset/weather/',     'weather.csv',      21,  'h', 'Weather'),
    ('custom', 'dataset/electricity/', 'electricity.csv',  321, 'h', 'ECL'),
    ('custom', 'dataset/traffic/',     'traffic.csv',      862, 'h', 'Traffic'),
    ('custom', 'dataset/exchange_rate/', 'exchange_rate.csv', 8, 'h', 'Exchange'),
]

# SOTA references from TimeMixer++ (Table 2) — for display only
SOTA_MSE = {
    ('ETTh1', 96): 0.355, ('ETTh1', 192): 0.394, ('ETTh1', 336): 0.430, ('ETTh1', 720): 0.447,
    ('ETTh2', 96): 0.274, ('ETTh2', 192): 0.336, ('ETTh2', 336): 0.329, ('ETTh2', 720): 0.384,
    ('ETTm1', 96): 0.312, ('ETTm1', 192): 0.356, ('ETTm1', 336): 0.384, ('ETTm1', 720): 0.440,
    ('ETTm2', 96): 0.163, ('ETTm2', 192): 0.217, ('ETTm2', 336): 0.273, ('ETTm2', 720): 0.360,
    ('Weather', 96): 0.146, ('Weather', 192): 0.193, ('Weather', 336): 0.244, ('Weather', 720): 0.315,
    ('ECL',  96): 0.129, ('ECL',  192): 0.148, ('ECL',  336): 0.165, ('ECL',  720): 0.202,
}

# 10% few-shot SOTA from TimeMixer++ (Table 5)
SOTA_FEWSHOT_MSE = {
    'ETT(Avg)': 0.396, 'Weather': 0.241, 'ECL': 0.168,
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='CAPE forecasting benchmark')

    # Data
    p.add_argument('--data_root', default=PKG_ROOT,
                   help='Root directory containing the dataset/ subdirectory')
    p.add_argument('--results_dir', default=os.path.join(PKG_ROOT, 'results'),
                   help='Where result .txt files are written')
    p.add_argument('--data',      default='ETTh1',
                   choices=[d[0] for d in ALL_DATASETS] + ['custom'],
                   help='Dataset key (used for single-dataset runs)')
    p.add_argument('--data_path', default='ETTh1.csv')
    p.add_argument('--root_path', default=os.path.join(PKG_ROOT, 'dataset', 'ETT-small'))
    p.add_argument('--enc_in',    type=int, default=7)
    p.add_argument('--freq',      default='h')

    # Forecasting
    p.add_argument('--seq_len',   type=int, default=720)
    p.add_argument('--pred_len',  type=int, default=96)
    p.add_argument('--label_len', type=int, default=48)
    p.add_argument('--percent',   type=int, default=100,
                   help='Percentage of training data (100 = full, 10 = few-shot)')

    # Model
    p.add_argument('--ridge_alpha',  type=float, default=1.0)
    p.add_argument('--period_len',   type=int,   default=-1,
                   help='Manual period (-1 = auto-detect)')
    p.add_argument('--max_period',   type=int,   default=-1,
                   help='Cap auto-detection (-1 = no cap)')
    p.add_argument('--min_period',   type=int,   default=2,
                   help='Floor for period auto-detection (1 = unconstrained; 2 reproduces the paper)')

    # Run modes
    p.add_argument('--sweep_alpha',   action='store_true',
                   help='Sweep ridge_alpha for a single dataset')
    p.add_argument('--table_full',    action='store_true',
                   help='Full-data sweep over all 7 datasets (Table 2)')
    p.add_argument('--table_fewshot', action='store_true',
                   help='10%%-few-shot sweep (Table 5)')

    # Misc
    p.add_argument('--batch_size',  type=int, default=64)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--no_cuda',     action='store_true')
    p.add_argument('--verbose',     action='store_true',
                   help='Print period detection and fit diagnostics')

    args = p.parse_args()
    _add_tslib_defaults(args)
    return args


def _add_tslib_defaults(args):
    """Extra attributes expected by tslib data_provider."""
    args.task_name          = 'long_term_forecast'
    args.features           = 'M'
    args.target             = 'OT'
    args.embed              = 'timeF'
    args.timeenc            = 1
    args.dec_in             = args.enc_in
    args.c_out              = args.enc_in
    args.augmentation_ratio = 0
    args.scale              = True
    args.inverse            = False
    args.cols               = None
    args.seasonal_patterns  = 'Monthly'
    args.use_multi_gpu      = False


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _make_args(base, data, root_path, data_path, enc_in, freq,
               percent=100, pred_len=None):
    a           = copy.deepcopy(base)
    a.data      = data
    a.root_path = root_path
    a.data_path = data_path
    a.enc_in    = enc_in
    a.dec_in    = enc_in
    a.c_out     = enc_in
    a.freq      = freq
    a.percent   = percent
    if pred_len is not None:
        a.pred_len = pred_len
    return a


def evaluate(model, test_loader, pred_len, device):
    """Return (MSE, MAE) on test set."""
    preds, trues = [], []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            bx, by = batch[0].float().to(device), batch[1].float().to(device)
            out    = model(bx)
            tgt    = by[:, -pred_len:, :]
            n      = min(out.shape[-1], tgt.shape[-1])
            preds.append(out[..., :n].cpu().numpy())
            trues.append(tgt[..., :n].cpu().numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    return float(np.mean((preds - trues) ** 2)), float(np.mean(np.abs(preds - trues)))


def run_one(args, device, alpha=None):
    """Fit + evaluate CAPE for a single (dataset, pred_len, alpha) combination."""
    if alpha is not None:
        args.ridge_alpha = alpha

    model = CAPE(args).to(device)
    _, train_loader = data_provider(args, 'train')
    _, test_loader  = data_provider(args, 'test')

    t0 = time.time()
    model.fit(train_loader, device=device)
    fit_time = time.time() - t0

    mse, mae = evaluate(model, test_loader, args.pred_len, device)
    return mse, mae, fit_time, model.period_len


# ---------------------------------------------------------------------------
# Single dataset, all pred_lens
# ---------------------------------------------------------------------------

def run_dataset(args, device, display='', percent=100, alpha=None):
    """Run all pred_lens for one dataset. Returns list of (pred_len, mse, mae)."""
    rows = []
    period = None
    for pl in PRED_LENS:
        a = copy.deepcopy(args)
        a.pred_len = pl
        a.percent  = percent
        mse, mae, _, period = run_one(a, device, alpha=alpha)
        rows.append((pl, mse, mae))
        sota = SOTA_MSE.get((display, pl))
        sota_str = f"  (sota {sota:.3f})" if sota else ""
        print(f"    H={pl:<4}  MSE={mse:.6f}  MAE={mae:.6f}{sota_str}")
    avg_mse = float(np.mean([r[1] for r in rows]))
    avg_mae = float(np.mean([r[2] for r in rows]))
    suffix = f"  (p={period})" if getattr(args, 'verbose', False) else ""
    print(f"    {'AVG':<5}  MSE={avg_mse:.6f}  MAE={avg_mae:.6f}{suffix}")
    return rows, avg_mse, avg_mae, period


# ---------------------------------------------------------------------------
# Table 2 — full-data sweep
# ---------------------------------------------------------------------------

def run_table_full(args, device):
    print("=" * 70)
    print("CAPE — Full-Data Forecasting (Table 2)")
    print(f"seq_len={args.seq_len}  ridge_alpha={args.ridge_alpha}")
    print("=" * 70)

    all_results = {}
    ett_mselist = []

    for (data, rp_rel, dp, enc_in, freq, display) in ALL_DATASETS:
        root_path = os.path.join(args.data_root, rp_rel)
        a = _make_args(args, data, root_path, dp, enc_in, freq)

        print(f"\n── {display} ──")
        rows, avg_mse, avg_mae, period = run_dataset(a, device, display=display)
        all_results[display] = (avg_mse, avg_mae)
        if display.startswith('ETT'):
            ett_mselist.append(avg_mse)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY  (avg over pred_lens 96/192/336/720)")
    print(f"  {'Dataset':<12}  {'AVG MSE':<10}  {'AVG MAE'}")
    print(f"  {'-'*38}")
    for name, (mse, mae) in all_results.items():
        print(f"  {name:<12}  {mse:<10.6f}  {mae:.6f}")
    if ett_mselist:
        print(f"  {'ETT(Avg)':<12}  {np.mean(ett_mselist):<10.6f}")
    print("=" * 70)

    os.makedirs(args.results_dir, exist_ok=True)
    out_path = os.path.join(args.results_dir, f'table_full_a{args.ridge_alpha}.txt')
    with open(out_path, 'w') as f:
        for name, (mse, mae) in all_results.items():
            f.write(f"{name}: mse={mse:.6f}, mae={mae:.6f}\n")
    print(f"\nSaved → {out_path}")


# ---------------------------------------------------------------------------
# Table 5 — 10% few-shot
# ---------------------------------------------------------------------------

FEWSHOT_DATASETS = ALL_DATASETS[:6]   # exclude Exchange (not in Table 5)

def run_table_fewshot(args, device):
    print("=" * 70)
    print("CAPE — Few-Shot Forecasting (10% training data) — Table 5")
    print(f"seq_len={args.seq_len}  ridge_alpha={args.ridge_alpha}")
    print("=" * 70)

    all_results = {}
    ett_mselist = []

    for (data, rp_rel, dp, enc_in, freq, display) in FEWSHOT_DATASETS:
        root_path = os.path.join(args.data_root, rp_rel)
        a = _make_args(args, data, root_path, dp, enc_in, freq, percent=10)

        print(f"\n── {display} (10% data) ──")
        rows, avg_mse, avg_mae, period = run_dataset(a, device, display=display, percent=10)
        all_results[display] = (avg_mse, avg_mae)
        if display.startswith('ETT'):
            ett_mselist.append(avg_mse)

    ett_avg = float(np.mean(ett_mselist)) if ett_mselist else float('nan')

    print("\n" + "=" * 70)
    print("TABLE 5 SUMMARY  (10% data, avg over pred_lens 96/192/336/720)")
    print(f"  {'Dataset':<12}  {'CAPE MSE':<12}  {'SOTA MSE':<12}  Delta")
    print(f"  {'-'*52}")
    print(f"  {'ETT(Avg)':<12}  {ett_avg:<12.6f}  "
          f"{SOTA_FEWSHOT_MSE['ETT(Avg)']:<12.3f}  "
          f"{ett_avg - SOTA_FEWSHOT_MSE['ETT(Avg)']:+.4f}")
    for name in ['Weather', 'ECL']:
        if name in all_results:
            mse, mae = all_results[name]
            sota = SOTA_FEWSHOT_MSE.get(name, float('nan'))
            sota_str = f"{sota:<12.3f}" if not np.isnan(sota) else "   —   "
            print(f"  {name:<12}  {mse:<12.6f}  {sota_str}  {mse - sota:+.4f}")
    print("=" * 70)

    os.makedirs(args.results_dir, exist_ok=True)
    out_path = os.path.join(args.results_dir, f'table_fewshot_a{args.ridge_alpha}.txt')
    with open(out_path, 'w') as f:
        f.write(f"ETT(Avg): mse={ett_avg:.6f}\n")
        for name, (mse, mae) in all_results.items():
            f.write(f"{name}: mse={mse:.6f}, mae={mae:.6f}\n")
    print(f"\nSaved → {out_path}")


# ---------------------------------------------------------------------------
# Ridge-alpha sweep
# ---------------------------------------------------------------------------

def run_sweep_alpha(args, device):
    alphas = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
    print(f"\nSweeping ridge_alpha for {args.data} H={args.pred_len}")
    head = f"  {'alpha':<10}  {'MSE':<12}  {'MAE':<12}"
    print(head + ('  period' if args.verbose else ''))
    print(f"  {'-'*46}")
    best_mse = float('inf')
    for a in alphas:
        mse, mae, _, period = run_one(copy.deepcopy(args), device, alpha=a)
        marker = ' *** BEST' if mse < best_mse else ''
        best_mse = min(best_mse, mse)
        col = f"  {period}" if args.verbose else ''
        print(f"  {a:<10}  {mse:<12.6f}  {mae:<12.6f}{col}{marker}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    device = torch.device(
        'cpu' if args.no_cuda else ('cuda' if torch.cuda.is_available() else 'cpu')
    )
    print(f"Device: {device}")

    if args.table_full:
        run_table_full(args, device)
    elif args.table_fewshot:
        run_table_fewshot(args, device)
    elif args.sweep_alpha:
        run_sweep_alpha(args, device)
    else:
        # Single run
        print(f"\nCAPE  {args.data}  H={args.pred_len}  "
              f"L={args.seq_len}  α={args.ridge_alpha}")
        mse, mae, fit_time, period = run_one(args, device)
        suffix = f"  period={period}" if args.verbose else ""
        print(f"  MSE={mse:.6f}  MAE={mae:.6f}  "
              f"fit_time={fit_time:.1f}s{suffix}")


if __name__ == '__main__':
    main()
