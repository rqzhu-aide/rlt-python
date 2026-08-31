"""Survival confidence bands (port of RLT's get.surv.band).

Requires a variance-enabled survival prediction. Two approaches,
mirroring the R package:

- "naive": marginal SDs = sqrt(diag(Cov)); MC band with full covariance.
- "smoothed": tensor-product penalized B-spline smoothing of the
  covariance surface (C++ ``rlt_tp_smooth`` — the mgcv ``te(i, j)``
  replacement), rank-k PSD truncation, positive-spectrum residual
  correction, eigenvalue-ratio weights — the same pipeline as R.
"""
from __future__ import annotations

import numpy as np

from ._core import mc_band, rlt_tp_smooth


def _nearest_psd(M):
    vals, vecs = np.linalg.eigh(M)
    vals = np.clip(vals, 0.0, None)
    Mp = (vecs * vals) @ vecs.T
    return (Mp + Mp.T) / 2.0


def _rank_k_psd(Sigma_psd, k_rank):
    p = Sigma_psd.shape[0]
    k = min(max(k_rank, 0), p)
    if k == 0:
        return np.zeros_like(Sigma_psd)
    vals, vecs = np.linalg.eigh(Sigma_psd)
    vals = np.clip(vals, 0.0, None)
    Uk = vecs[:, -k:]                      # eigh ascending -> top-k last
    lamk = vals[-k:]
    SigK = (Uk * lamk) @ Uk.T
    return (SigK + SigK.T) / 2.0


def _pos_spectrum_psd(M):
    vals, vecs = np.linalg.eigh(M)
    pos = vals > 0
    if not pos.any():
        return np.zeros_like(M)
    Mp = (vecs[:, pos] * vals[pos]) @ vecs[:, pos].T
    return (Mp + Mp.T) / 2.0


def get_surv_band(model, X, i=0, alpha=0.05, approach="naive", nsim=5000,
                  k_rank=10, k_mode="fixed", k_prop=0.99, seed=0,
                  band_grid_size=0):
    """Simultaneous confidence bands for survival curves.

    Parameters
    ----------
    model : fitted RLTSurvivalForest with var_mode != "none"
    X : test covariates
    i : subject index (0 = all); 1-based like R when positive
    alpha : level or array of levels
    approach : {"naive", "smoothed"}
    nsim : MC simulations for the critical value
    k_rank / k_mode / k_prop : rank truncation controls ("smoothed")
    seed : RNG seed for the MC band (deterministic given seed)
    band_grid_size : optional reduced grid (see predict_var)

    Returns
    -------
    dict with ``lower``, ``upper`` (n, T), ``timepoints``, per-subject
    ``diagnostics``, and ``marsd``.
    """
    survival, cov = model.predict_var(
        X, band_grid_size=band_grid_size)
    chf = -np.log(np.clip(survival, 1e-12, None))
    timepoints = model.predict(X, band_grid_size=band_grid_size)
    # NOTE: timepoints for the grid come from the model's failure times
    tp_grid = _band_grid(model, band_grid_size)

    N = cov.shape[0]
    alphas = np.atleast_1d(np.asarray(alpha, dtype=np.float64))

    if np.any(alphas < 0) or np.any(alphas > 0.5):
        raise ValueError("alpha must be in [0, 0.5]")

    if i == 0:
        allid = np.arange(N)
    else:
        allid = np.atleast_1d(np.asarray(i, dtype=int)) - 1  # 1-based
        if np.any(allid < 0) or np.any(allid >= N):
            raise ValueError(f"observation {i} does not exist")

    eps = np.finfo(float).eps
    lowers, uppers, diags = [], [], []

    for idx in allid:
        Sigma_raw = np.asarray(cov[idx], dtype=np.float64)
        raw_sd = np.sqrt(np.maximum(np.diag(Sigma_raw), eps))

        if approach == "naive":
            marsd = raw_sd.copy()
            bandk = np.atleast_1d(
                np.asarray(mc_band(marsd, Sigma_raw, alphas, nsim, seed),
                            dtype=np.float64))
            diag_info = {"timepoint": tp_grid, "raw_sd": raw_sd,
                         "final_sd": marsd}
            ve_info = {"var_explained": 1.0, "k_rank": None,
                       "total_var": float(np.trace(Sigma_raw)),
                       "lowrank_var": float(np.trace(Sigma_raw))}
            eig_info = None
        elif approach == "smoothed":
            if k_mode not in ("fixed", "proportion"):
                raise ValueError("k_mode must be 'fixed' or 'proportion'")
            if k_mode == "proportion":
                if not 0 < k_prop <= 1:
                    raise ValueError("k_prop must be in (0, 1]")
                vals = np.clip(
                    np.linalg.eigvalsh(_nearest_psd(Sigma_raw)), 0, None)
                total = vals.sum()
                if total <= eps:
                    k_used = 0
                else:
                    cum = np.cumsum(vals[::-1]) / total
                    k_used = int(np.searchsorted(cum, k_prop) + 1)
                    k_used = min(k_used, vals.shape[0])
            else:
                k_used = int(k_rank)

            # C++ tensor-product penalized B-spline smoother (GCV lambda)
            Sigma_s = np.asarray(rlt_tp_smooth(
                Sigma_raw, tp_grid.astype(np.float64).ravel(),
                max(k_used, 4), -1.0))
            Sigma_psd = _nearest_psd(Sigma_s)
            Sigma_smoothK = _rank_k_psd(Sigma_psd, k_used) \
                if k_used > 0 else np.zeros_like(Sigma_psd)
            smoothK_sd = np.sqrt(np.maximum(np.diag(Sigma_smoothK), eps))

            R = (Sigma_raw - Sigma_smoothK)
            R = (R + R.T) / 2.0
            R_pos = _pos_spectrum_psd(R)
            sdB = np.sqrt(max(float(np.diag(R_pos).max()), 0.0))

            tr_smoothK = float(np.clip(np.diag(Sigma_smoothK), 0, None).sum())
            tr_R_pos = float(np.clip(np.diag(R_pos), 0, None).sum())
            tr_total = tr_smoothK + tr_R_pos
            w_k = tr_smoothK / tr_total if tr_total > eps else 1.0
            w_r = 1.0 - w_k if tr_total > eps else 0.0

            marsd = np.sqrt(np.maximum(
                w_k * smoothK_sd ** 2 + w_r * sdB ** 2, eps))
            bandk = np.atleast_1d(
                np.asarray(mc_band(marsd, Sigma_raw, alphas, nsim, seed),
                            dtype=np.float64))
            diag_info = {"timepoint": tp_grid, "raw_sd": raw_sd,
                         "smoothK_sd": smoothK_sd, "final_sd": marsd,
                         "resid_sd": np.full(tp_grid.shape, sdB)}
            ve_info = {
                "var_explained":
                    float(np.trace(Sigma_smoothK) / np.trace(Sigma_raw))
                    if np.trace(Sigma_raw) > 0 else 0.0,
                "k_rank": int(k_used),
                "total_var": float(np.trace(Sigma_raw)),
                "lowrank_var": tr_smoothK,
                "resid_var": float(np.trace(Sigma_raw) - tr_smoothK),
            }
            eig_info = {"k_used": int(k_used), "k_prop": float(k_prop)}
        else:
            raise ValueError(f"approach {approach!r} not available")

        k_crit = bandk[0] if bandk.shape[0] == 1 else bandk
        lowers.append(np.exp(-chf[idx] - k_crit))
        uppers.append(np.exp(-chf[idx] + k_crit))
        diags.append({"diag_sd": diag_info, "var_explained": ve_info,
                      "eig": eig_info})

    return {
        "lower": np.asarray(lowers),
        "upper": np.asarray(uppers),
        "timepoints": tp_grid,
        "marsd": None,
        "diagnostics": diags,
    }


def _band_grid(model, band_grid_size):
    """Time grid actually used by predict_var for the given size."""
    original = model.timepoints_
    nfail = original.shape[0]
    bgs = int(band_grid_size or 0)
    if bgs > 0 and bgs < nfail:
        probs = np.linspace(0.05, 1.0, min(bgs, nfail))
        grid = np.unique(
            np.quantile(original, probs, method="nearest"))
        return np.sort(grid).astype(np.float64)
    return np.asarray(original, dtype=np.float64)
