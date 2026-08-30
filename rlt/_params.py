"""Internal: shared parameter handling for rlt estimators."""

from __future__ import annotations

from ._core import CoreParams

# public sklearn-ish names -> CoreParams attribute names
_PARAM_MAP = {
    "n_estimators": "ntrees",
    "min_samples_leaf": "nmin",
    "mtry": "mtry",
    "nsplit": "nsplit",
    "resample_replace": "replacement",
    "resample_prob": "resample_prob",
    "importance": "importance",
    "resample_track": "obs_track",
    "var_mode": "var_mode",
    "linear_comb": "linear_comb",
    "linear_comb_method": "linear_comb_method",
    "split_rule": "split_rule",
    "alpha": "alpha",
    "embed_ntrees": "embed_ntrees",
    "embed_mtry": "embed_mtry",
    "embed_nmin": "embed_nmin",
    "embed_nsplit": "embed_nsplit",
    "embed_resample_replace": "embed_replacement",
    "embed_resample_prob": "embed_resample_prob",
    "embed_mute": "embed_mute",
    "embed_protect": "embed_protect",
    "embed_threshold": "embed_threshold",
    "reinforcement": "reinforcement",
}

# importance: "none"/"permute"/"distribute" -> 0/1/2
_IMPORTANCE_CODES = {"none": 0, "permute": 1, "distribute": 2}

# var_mode: "none"/"matched"/"ij"/"jack" -> 0/1/2/3
_VAR_MODE_CODES = {"none": 0, "matched": 1, "ij": 2, "jack": 3}


def _encode(value, table, name):
    if value is None:
        return None
    if isinstance(value, bool):
        value = int(value)
    if isinstance(value, str):
        key = value.lower()
        if key not in table:
            raise ValueError(
                f"{name}={value!r} not recognized; choose from {sorted(table)}"
            )
        return table[key]
    return int(value)


def build_core_params(estimator, n: int, p: int, seed: int) -> CoreParams:
    """Translate an estimator's public hyper-parameters to CoreParams."""
    cp = CoreParams()
    cp.n = int(n)
    cp.p = int(p)

    imp = _encode(estimator.importance, _IMPORTANCE_CODES, "importance")
    if imp is None:
        imp = 0
    if imp == 1 and estimator.var_mode:
        imp = 2  # permute is incompatible with var.mode; use distribute

    vm = _encode(estimator.var_mode, _VAR_MODE_CODES, "var_mode")
    if vm is None:
        vm = 0

    values = {
        "ntrees": int(estimator.n_estimators),
        "nmin": int(estimator.min_samples_leaf),
        "mtry": int(estimator.mtry if estimator.mtry is not None else max(1, p // 2)),
        "nsplit": int(estimator.nsplit),
        "replacement": bool(estimator.resample_replace),
        "resample_prob": float(
            estimator.resample_prob
            if estimator.resample_prob is not None
            else (1.0 if estimator.resample_replace else 0.8)
        ),
        "use_obs_w": False,  # set by fit() when sample_weight is given
        "use_var_prob": False,  # set by fit() when var_prob is given
        "importance": imp,
        "reinforcement": bool(estimator.reinforcement),
        "obs_track": bool(estimator.resample_track) or vm > 0,
        "var_mode": vm,
        "linear_comb": int(estimator.linear_comb),
        "alpha": float(estimator.alpha),
        "split_rule": 1,
        "linear_comb_method": 1,
        "embed_ntrees": int(estimator.embed_ntrees),
        "embed_mtry": float(estimator.embed_mtry),
        "embed_nmin": int(estimator.embed_nmin),
        "embed_nsplit": int(estimator.embed_nsplit),
        "embed_replacement": bool(estimator.embed_resample_replace),
        "embed_resample_prob": float(estimator.embed_resample_prob),
        "embed_mute": float(estimator.embed_mute),
        "embed_protect": int(estimator.embed_protect),
        "embed_threshold": float(estimator.embed_threshold),
        "ncores": int(estimator.n_jobs if estimator.n_jobs and estimator.n_jobs > 0 else 0),
        "verbose": int(estimator.verbose),
        "seed": int(seed),
    }
    for k, v in values.items():
        setattr(cp, k, v)
    return cp


def check_var_prob(var_prob, p):
    import numpy as np

    vp = np.asarray(var_prob, dtype=np.float64).ravel()
    if vp.shape[0] != p:
        raise ValueError(f"var_prob must have length p={p}, got {vp.shape[0]}")
    if np.any(vp < 0):
        raise ValueError("var_prob cannot be negative")
    s = vp.sum()
    if s <= 0:
        raise ValueError("var_prob must contain at least one positive weight")
    return vp / s


def check_obs_weight(sample_weight, n):
    import numpy as np

    w = np.asarray(sample_weight, dtype=np.float64).ravel()
    if w.shape[0] != n:
        raise ValueError(f"sample_weight must have length n={n}, got {w.shape[0]}")
    if np.any(w < 0):
        raise ValueError("sample_weight cannot be negative")
    s = w.sum()
    if s <= 0:
        raise ValueError("sample_weight must contain at least one positive weight")
    return w / s
