"""
CAPE — Cross-Dataset Zero-Shot Evaluation Script

Reproduces Table 6 from the paper: train CAPE on one ETT dataset,
evaluate on a *different* ETT dataset without any fine-tuning.

This works because CAPE's operator W lives in segment space (dimension S = L/p),
which is independent of the number of channels. When the target has the same
number of channels as the source, W transfers directly.

Quick start
-----------
# Full Table 6 sweep (6 source→target pairs, avg over pred_lens):
    python run_zeroshot.py --table6

# Single pair:
    python run_zeroshot.py \
        --src_data ETTh1 --src_path dataset/ETT-small/ --src_file ETTh1.csv \
        --tgt_data ETTh2 --tgt_path dataset/ETT-small/ --tgt_file ETTh2.csv \
        --enc_in 7 --pred_len 96

Results reported in the paper (avg over pred_lens 96/192/336/720)
-----------------------------------------------------------------
  ETTh1 → ETTh2 :  MSE = 0.349
  ETTh1 → ETTm2 :  MSE = 0.295
  ETTm1 → ETTm2 :  MSE = 0.256
  ETTm1 → ETTh2 :  MSE = 0.355
  ETTh2 → ETTh1 :  MSE = 0.417
  ETTm2 → ETTm1 :  MSE = 0.404
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
# Table 6 pairs: (src_data, src_root, src_file, enc_in, src_freq,
#                 tgt_data, tgt_root, tgt_file, enc_in, tgt_freq)
# ---------------------------------------------------------------------------

TABLE6_PAIRS = [
    # Strong cross-domain pairs (different temporal resolution)
    ('ETTh1', 'dataset/ETT-small/', 'ETTh1.csv', 7, 'h',
     'ETTh2', 'dataset/ETT-small/', 'ETTh2.csv', 7, 'h'),

    ('ETTh1', 'dataset/ETT-small/', 'ETTh1.csv', 7, 'h',
     'ETTm2', 'dataset/ETT-small/', 'ETTm2.csv', 7, 'm'),

    ('ETTh2', 'dataset/ETT-small/', 'ETTh2.csv', 7, 'h',
     'ETTh1', 'dataset/ETT-small/', 'ETTh1.csv', 7, 'h'),

    ('ETTm1', 'dataset/ETT-small/', 'ETTm1.csv', 7, 'm',
     'ETTh2', 'dataset/ETT-small/', 'ETTh2.csv', 7, 'h'),

    ('ETTm1', 'dataset/ETT-small/', 'ETTm1.csv', 7, 'm',
     'ETTm2', 'dataset/ETT-small/', 'ETTm2.csv', 7, 'm'),

    ('ETTm2', 'dataset/ETT-small/', 'ETTm2.csv', 7, 'm',
     'ETTm1', 'dataset/ETT-small/', 'ETTm1.csv', 7, 'm'),
]

PRED_LENS = [96, 192, 336, 720]

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='CAPE cross-dataset zero-shot (Table 6)')

    p.add_argument('--data_root', default=PKG_ROOT,
                   help='Root directory containing the dataset/ folder')
    p.add_argument('--results_dir', default=os.path.join(PKG_ROOT, 'results'),
                   help='Where result .txt files are written')

    # Single-pair run
    p.add_argument('--src_data', default='ETTh1')
    p.add_argument('--src_path', default='dataset/ETT-small/')
    p.add_argument('--src_file', default='ETTh1.csv')
    p.add_argument('--tgt_data', default='ETTh2')
    p.add_argument('--tgt_path', default='dataset/ETT-small/')
    p.add_argument('--tgt_file', default='ETTh2.csv')
    p.add_argument('--enc_in',   type=int, default=7,
                   help='Number of channels (must match between src and tgt)')
    p.add_argument('--src_freq', default='h')
    p.add_argument('--tgt_freq', default='h')
    p.add_argument('--pred_len', type=int, default=96)

    # Full table
    p.add_argument('--table6', action='store_true',
                   help='Run all 6 source→target pairs (avg over pred_lens)')

    # Model
    p.add_argument('--seq_len',     type=int,   default=720)
    p.add_argument('--label_len',   type=int,   default=48)
    p.add_argument('--ridge_alpha', type=float, default=1.0)
    p.add_argument('--period_len',  type=int,   default=-1)
    p.add_argument('--max_period',  type=int,   default=-1)
    p.add_argument('--min_period',  type=int,   default=2,
                   help='Floor for period auto-detection (1 = unconstrained; 2 reproduces the paper)')

    p.add_argument('--batch_size',  type=int, default=64)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--no_cuda',     action='store_true')
    p.add_argument('--verbose',     action='store_true',
                   help='Print period detection and fit diagnostics')

    args = p.parse_args()
    _add_tslib_defaults(args)
    return args


def _add_tslib_defaults(args):
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
    args.percent            = 100


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_args(base, data, root_path, data_path, enc_in, freq, pred_len=None):
    a           = copy.deepcopy(base)
    a.data      = data
    a.root_path = root_path
    a.data_path = data_path
    a.enc_in    = enc_in
    a.dec_in    = enc_in
    a.c_out     = enc_in
    a.freq      = freq
    if pred_len is not None:
        a.pred_len = pred_len
    return a


def evaluate(model, test_loader, pred_len, device):
    preds, trues = [], []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            bx = batch[0].float().to(device)
            by = batch[1].float().to(device)
            out = model(bx)
            tgt = by[:, -pred_len:, :]
            n   = min(out.shape[-1], tgt.shape[-1])
            preds.append(out[..., :n].cpu().numpy())
            trues.append(tgt[..., :n].cpu().numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    return float(np.mean((preds - trues) ** 2)), float(np.mean(np.abs(preds - trues)))


def run_pair(src_args, tgt_args, device, alpha=None):
    """
    Train on src, evaluate on tgt (zero-shot transfer).

    CAPE's W is in segment space R^{S×M} where S = seq_len / period.
    This is channel-agnostic, so W transfers directly to tgt as long as
    seq_len matches and enc_in matches (same physical series structure).
    """
    if alpha is not None:
        src_args.ridge_alpha = alpha

    model = CAPE(src_args).to(device)
    _, train_loader = data_provider(src_args, 'train')

    t0 = time.time()
    model.fit(train_loader, device=device)
    fit_time = time.time() - t0

    _, test_loader = data_provider(tgt_args, 'test')
    mse, mae = evaluate(model, test_loader, tgt_args.pred_len, device)
    return mse, mae, fit_time, model.period_len


# ---------------------------------------------------------------------------
# Table 6
# ---------------------------------------------------------------------------

def run_table6(args, device):
    print("=" * 70)
    print("CAPE — Cross-Dataset Zero-Shot Forecasting (Table 6)")
    print(f"seq_len={args.seq_len}  ridge_alpha={args.ridge_alpha}")
    print("Averaged over pred_lens: 96 / 192 / 336 / 720")
    print("=" * 70)

    summary = []   # (label, avg_mse, avg_mae)

    for (sd, sr, sc, se, sf, td, tr, tc, te, tf) in TABLE6_PAIRS:
        src_root = os.path.join(args.data_root, sr)
        tgt_root = os.path.join(args.data_root, tr)
        label    = f"{sd} → {td}"
        print(f"\n── {label} ──")
        print(f"  {'H':<6}  {'MSE':<12}  {'MAE'}")
        print(f"  {'-'*32}")

        rows = []
        for pl in PRED_LENS:
            s_a = _make_args(args, sd, src_root, sc, se, sf, pred_len=pl)
            t_a = _make_args(args, td, tgt_root, tc, te, tf, pred_len=pl)
            mse, mae, ft, period = run_pair(s_a, t_a, device)
            rows.append((pl, mse, mae))
            info = (f"  (p={period}, fit={ft:.1f}s)" if args.verbose
                    else f"  (fit={ft:.1f}s)")
            print(f"  {pl:<6}  {mse:<12.6f}  {mae:.6f}{info}")

        avg_mse = float(np.mean([r[1] for r in rows]))
        avg_mae = float(np.mean([r[2] for r in rows]))
        print(f"  {'AVG':<6}  {avg_mse:<12.6f}  {avg_mae:.6f}")
        summary.append((label, avg_mse, avg_mae))

    print("\n" + "=" * 70)
    print("TABLE 6 SUMMARY  (avg over pred_lens 96/192/336/720)")
    print(f"  {'Pair':<22}  {'AVG MSE':<12}  {'AVG MAE'}")
    print(f"  {'-'*46}")
    for label, mse, mae in summary:
        print(f"  {label:<22}  {mse:<12.6f}  {mae:.6f}")
    print("=" * 70)

    os.makedirs(args.results_dir, exist_ok=True)
    out_path = os.path.join(args.results_dir, f'table6_zeroshot_a{args.ridge_alpha}.txt')
    with open(out_path, 'w') as f:
        for label, mse, mae in summary:
            f.write(f"{label}: avg_mse={mse:.6f}, avg_mae={mae:.6f}\n")
    print(f"\nSaved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    device = torch.device(
        'cpu' if args.no_cuda else ('cuda' if torch.cuda.is_available() else 'cpu')
    )
    print(f"Device: {device}")

    if args.table6:
        run_table6(args, device)
    else:
        # Single pair
        src_root = os.path.join(args.data_root, args.src_path)
        tgt_root = os.path.join(args.data_root, args.tgt_path)
        src_args = _make_args(args, args.src_data, src_root, args.src_file,
                              args.enc_in, args.src_freq, pred_len=args.pred_len)
        tgt_args = _make_args(args, args.tgt_data, tgt_root, args.tgt_file,
                              args.enc_in, args.tgt_freq, pred_len=args.pred_len)

        label = f"{args.src_data} → {args.tgt_data}"
        print(f"\nCAPE zero-shot:  {label}  H={args.pred_len}")
        mse, mae, fit_time, period = run_pair(src_args, tgt_args, device)
        suffix = f"  period={period}" if args.verbose else ""
        print(f"  MSE={mse:.6f}  MAE={mae:.6f}  "
              f"fit_time={fit_time:.1f}s{suffix}")


if __name__ == '__main__':
    main()
