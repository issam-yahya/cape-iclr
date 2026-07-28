"""
CAPE: Closed-form Analytic Period-folded Estimation

A zero-parameter time-series forecasting model.

Core idea
---------
Given a lookback window x ∈ R^{L×C}:

1. **Period detection** (FFT + autocorrelation): find period p that divides L.
2. **Instance normalisation**: subtract per-sequence mean.
3. **Period folding**: reshape x → X ∈ R^{p × S}  (S = L/p segments).
4. **Ridge regression** (analytical): solve
       W = (A^T A + λ I)^{-1} A^T B
   where A = X (flattened) and B = Y (folded target).
5. **Forecast**: Ŷ = X @ W, unfolded and truncated to pred_len.

W is shared across all channels (channel-agnostic), and is fitted by
accumulating the normal equations A^T A, A^T B over all training windows —
a single closed-form solve, no gradient descent, no learned parameters.

Cross-dataset zero-shot
-----------------------
Because W operates in *segment space* (dimension S = L/p), it is independent
of the number of channels. You can train on source data and apply directly to
a target dataset with a different channel count.

Usage (Python API)
------------------
    from cape import CAPE
    import torch

    class Cfg:
        seq_len = 720; pred_len = 96; enc_in = 7; ridge_alpha = 1.0

    model = CAPE(Cfg())
    model.fit(train_loader, device='cuda')
    pred = model(batch_x)               # [B, pred_len, C]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import gcd, ceil


class CAPE(nn.Module):
    """
    CAPE: zero-parameter time-series forecasting.

    Config attributes read
    ----------------------
    seq_len      : int   — lookback window length
    pred_len     : int   — forecast horizon
    enc_in       : int   — number of input channels
    ridge_alpha  : float — ridge regularisation λ  (default 1.0)
    period_len   : int   — manual period override; -1 = auto-detect (default -1)
    max_period   : int   — cap auto-detection to this value; -1 = no cap (default -1)
    min_period   : int   — floor for auto-detection (default 2, which reproduces
                           the paper; 8 = the original hardcoded MIN_PERIOD)
    task_name    : str   — 'long_term_forecast' or 'anomaly_detection' (default forecast)
    verbose      : bool  — print period detection / fit diagnostics (default False)
    """

    def __init__(self, configs):
        super().__init__()

        self.task_name  = getattr(configs, 'task_name',   'long_term_forecast')
        self.seq_len    = configs.seq_len
        self.pred_len   = getattr(configs, 'pred_len',    0)
        self.enc_in     = configs.enc_in
        self.ridge_alpha = getattr(configs, 'ridge_alpha', 1.0)
        self.manual_period = getattr(configs, 'period_len', -1)
        self.max_period    = getattr(configs, 'max_period', -1)
        self.min_period    = getattr(configs, 'min_period', 2)
        self.verbose       = getattr(configs, 'verbose', False)

        # Set during fit()
        self.period_len = None
        self.seg_num_x  = None
        self.seg_num_y  = None
        self.out_len    = None

        # Buffers for the fitted operator W and a fitted flag
        self.register_buffer('W',      torch.zeros(1))
        self.register_buffer('fitted', torch.tensor(False))

        # Dummy parameter so the model is compatible with optimisers / .parameters()
        self.bias = nn.Parameter(torch.zeros(1))

    # ------------------------------------------------------------------
    # Period detection
    # ------------------------------------------------------------------

    def _valid_periods(self):
        """Return candidate periods that divide seq_len with enough segments."""
        # Lower bound on p. This is a *floor*, not a cap — with max_period = -1
        # the search already runs all the way up to seq_len. It defaults to 2:
        # p = 1 is the degenerate case where folding is a no-op and CAPE reduces
        # to plain ridge regression on the raw seq_len-dim lookback.
        min_p = max(1, self.min_period)
        cap = self.max_period if self.max_period > 0 else self.seq_len
        out = []
        for p in range(min_p, min(self.seq_len, cap) + 1):
            if self.seq_len % p != 0:
                continue
            seg_x = self.seq_len // p
            if self.task_name == 'anomaly_detection':
                if seg_x >= 3:
                    out.append(p)
            else:
                seg_y = ceil(self.pred_len / p)
                if seg_x >= 2 and seg_y >= 1:
                    out.append(p)
        return out

    def _detect_period(self, x_agg):
        """
        Score candidate periods with FFT power + autocorrelation.

        x_agg : [L] — aggregated 1-D signal (e.g. channel mean over training set)
        Returns the best-scoring period p.
        """
        x = x_agg - x_agg.mean()
        candidates = self._valid_periods()
        if not candidates:
            return max(2, gcd(self.seq_len, max(self.pred_len, 1)))

        # FFT power spectrum
        try:
            fft_v = torch.fft.rfft(x)
            power = fft_v.abs() ** 2
        except AttributeError:                        # PyTorch < 1.8
            fft_v = torch.rfft(x, 1)
            power = fft_v[:, 0] ** 2 + fft_v[:, 1] ** 2
        power[0] = 0                                  # remove DC component

        norm   = (x ** 2).sum().clamp(min=1e-10)
        maxpow = power.max().item() + 1e-10

        scores = {}
        for p in candidates:
            # Fundamental frequency + 2 harmonics
            base = round(self.seq_len / p)
            fft_s = sum(
                power[base * h].item()
                for h in range(1, 4)
                if 0 < base * h < len(power)
            )
            acf_s = float((x[:-p] * x[p:]).sum() / norm)
            scores[p] = 0.5 * fft_s / (maxpow * 3) + 0.5 * max(0.0, acf_s)

        best = max(scores, key=scores.get)
        if self.verbose:
            top5 = sorted(scores.items(), key=lambda kv: -kv[1])[:5]
            print(f"  [CAPE] Period detection: {len(candidates)} candidates, "
                  f"top-5 = {top5}")
            print(f"  [CAPE] Selected period p = {best}")
        return best

    # ------------------------------------------------------------------
    # Internal transforms
    # ------------------------------------------------------------------

    def _smooth(self, x):
        """Moving-average smoothing along the time axis. x: [N, 1, L]."""
        k = 1 + 2 * (self.period_len // 2)
        pad = self.period_len // 2
        kernel = torch.ones(1, 1, k, device=x.device) / k
        return F.conv1d(x, kernel, padding=pad)

    def _fold_x(self, x):
        """
        Normalise, smooth, and reshape x by the detected period.

        x : [B, L, C]
        Returns
        -------
        x_folded : [B*C, p, S]  where p = period_len, S = seg_num_x
        seq_mean : [B, 1, C]
        B        : int
        """
        B = x.shape[0]
        seq_mean = x.mean(dim=1, keepdim=True)         # [B, 1, C]
        x = (x - seq_mean).permute(0, 2, 1)            # [B, C, L]

        C = x.shape[1]
        x = self._smooth(x.reshape(-1, 1, self.seq_len)).reshape(-1, C, self.seq_len) + x

        x = x.reshape(-1, self.seg_num_x, self.period_len)
        x = x.permute(0, 2, 1)                         # [B*C, p, S]
        return x, seq_mean, B

    def _fold_y(self, x_enc, y):
        """
        Fold the target y using x_enc's mean for normalisation.

        y : [B, H, C] — raw target
        Returns [B*C, p, seg_num_y]
        """
        mean = x_enc.mean(dim=1, keepdim=True)
        y = y - mean                                    # [B, H, C]

        # Zero-pad if pred_len is not divisible by period_len
        if self.out_len > self.pred_len:
            pad = self.out_len - self.pred_len
            y = F.pad(y, (0, 0, 0, pad))

        y = y.permute(0, 2, 1)                         # [B, C, out_len]
        y = y.reshape(-1, self.seg_num_y, self.period_len)
        y = y.permute(0, 2, 1)                         # [B*C, p, seg_num_y]
        return y

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, train_loader, device='cpu'):
        """
        Fit CAPE analytically on the full training set.

        Accumulates A^T A and A^T B over all batches, then solves
            W = (A^T A + λ I)^{-1} A^T B
        in one closed-form step. No iterations, no gradients.

        Parameters
        ----------
        train_loader : DataLoader yielding (x, y, ...) batches
        device       : torch.device or str
        """
        self.eval()

        # ── Step 1: Period detection ─────────────────────────────────
        if self.manual_period > 0:
            self.period_len = self.manual_period
            if self.verbose:
                print(f"  [CAPE] Using manual period p = {self.period_len}")
        else:
            if self.verbose:
                print("  [CAPE] Detecting period from training data …")
            x_acc = []
            for batch in train_loader:
                bx = batch[0].float().to(device)
                x_acc.append(bx.permute(0, 2, 1).reshape(-1, self.seq_len))
                if sum(a.shape[0] for a in x_acc) > 10_000:
                    break
            x_all = torch.cat(x_acc, dim=0)
            self.period_len = self._detect_period(x_all.mean(dim=0))

        # ── Step 2: Dimensions ───────────────────────────────────────
        p = self.period_len
        self.seg_num_x = self.seq_len // p
        if self.task_name == 'anomaly_detection':
            self.seg_num_y = self.seg_num_x
            self.out_len   = self.seq_len
        else:
            self.seg_num_y = ceil(self.pred_len / p)
            self.out_len   = self.seg_num_y * p

        S = self.seg_num_x
        M = self.seg_num_y

        if self.task_name == 'anomaly_detection':
            K = S - 1
            AtA = torch.zeros(K, K, device=device)
            AtB = torch.zeros(K, K, device=device)
        else:
            AtA = torch.zeros(S, S, device=device)
            AtB = torch.zeros(S, M, device=device)

        n_rows = 0

        # ── Step 3: Accumulate normal equations ─────────────────────
        with torch.no_grad():
            for batch in train_loader:
                bx = batch[0].float().to(device)
                x_folded, _, _ = self._fold_x(bx)

                if self.task_name == 'anomaly_detection':
                    K = S - 1
                    A = x_folded[:, :, :-1].reshape(-1, K)
                    B = x_folded[:, :,  1:].reshape(-1, K)
                else:
                    by = batch[1].float().to(device)
                    tgt = by[:, -self.pred_len:, :]
                    y_folded = self._fold_y(bx, tgt)
                    A = x_folded.reshape(-1, S)
                    B = y_folded.reshape(-1, M)

                AtA   += A.T @ A
                AtB   += A.T @ B
                n_rows += A.shape[0]

        # ── Step 4: Solve ────────────────────────────────────────────
        if self.verbose:
            print(f"  [CAPE] Accumulated {n_rows} sample-rows")
        dim = (S - 1) if self.task_name == 'anomaly_detection' else S
        reg = self.ridge_alpha * torch.eye(dim, device=device)

        try:
            self.W = torch.linalg.solve(AtA + reg, AtB)
        except AttributeError:                          # PyTorch < 1.8
            try:
                self.W, _ = torch.solve(AtB, AtA + reg)
            except Exception:
                self.W = torch.pinverse(AtA + reg) @ AtB
        except Exception:
            self.W = torch.pinverse(AtA + reg) @ AtB

        self.fitted = torch.tensor(True, device=device)
        if self.verbose:
            print(f"  [CAPE] W shape: {self.W.shape},  ‖W‖ = {self.W.norm():.4f}")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _forecast(self, x):
        """x: [B, L, C] → [B, pred_len, C]"""
        B = x.shape[0]
        if not self.fitted:
            return torch.zeros(B, self.pred_len, self.enc_in,
                               device=x.device) + self.bias * 0

        x_folded, seq_mean, B = self._fold_x(x)          # [B*C, p, S]
        y = x_folded @ self.W                             # [B*C, p, M]
        C = x_folded.shape[0] // B

        y = y.permute(0, 2, 1).reshape(B, C, self.out_len)
        y = y[:, :, :self.pred_len]                       # truncate to H
        return y.permute(0, 2, 1) + seq_mean + self.bias * 0   # [B, H, C]

    def _anomaly_detect(self, x):
        """x: [B, L, C] → [B, L, C] reconstructed (one-step-ahead in segment space)"""
        B = x.shape[0]
        if not self.fitted:
            return x + self.bias * 0

        x_folded, seq_mean, B = self._fold_x(x)
        S = self.seg_num_x
        K = S - 1

        x_in   = x_folded[:, :, :-1]                     # [B*C, p, K]
        x_pred = x_in.reshape(-1, K) @ self.W
        x_pred = x_pred.reshape(x_in.shape)               # [B*C, p, K]

        x_recon = torch.cat([x_folded[:, :, :1], x_pred], dim=2)  # [B*C, p, S]
        C = x_folded.shape[0] // B
        x_recon = x_recon.permute(0, 2, 1).reshape(B, C, self.seq_len)
        return x_recon.permute(0, 2, 1) + seq_mean + self.bias * 0

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name == 'anomaly_detection':
            return self._anomaly_detect(x_enc)
        return self._forecast(x_enc)
