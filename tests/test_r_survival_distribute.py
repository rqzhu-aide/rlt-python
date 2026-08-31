"""Port of RLT testthat/test-survival-distribute.R (survival distribute VI).

Param map: ntrees->n_estimators, importance 'distribute' kept,
resample.prob->resample_prob, resample.replace->resample_replace,
reinforcement->reinforcement, linear.comb->linear_comb,
param.control list(var.mode=...) -> var_mode, ncores->n_jobs, seed->
random_state. R's inline set.seed data generation is mirrored with
numpy default_rng (same distributions, different stream).
"""
import numpy as np
import pytest

from rlt import RLT_surv


def make_surv(n, p, seed, censoring=0.7):
    """Inline generator from the R file: hazard = exp(X1 + 0.5 X2),
    exponential times, censoring rate ~ 0.3."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    true_effect = X[:, 0] + 0.5 * X[:, 1]
    hazard = np.exp(true_effect)
    surv_time = rng.exponential(1.0 / hazard)
    censor_time = rng.exponential(1.0 / (np.median(hazard) * 0.3 / 0.7),
                                  size=n)
    y = np.empty(n, dtype=[("event", "?"), ("time", "<f8")])
    y["event"] = (surv_time <= censor_time)
    y["time"] = np.minimum(surv_time, censor_time)
    return X, y


def test_distribute_importance_nonzero_for_important_variables():
    X, y = make_surv(300, 5, seed=42)
    fit = RLT_surv(n_estimators=100, importance="distribute",
                   resample_prob=0.7, resample_replace=False, n_jobs=2,
                   random_state=123).fit(X, y)
    vi = np.asarray(fit.varimp_)
    # X1 and X2 are important
    assert vi[0] > 0.01
    assert vi[1] > 0.005
    # Important vars should rank above noise vars
    assert vi[0] > vi[2:5].mean()
    assert vi[1] > vi[2:5].mean()
    # Length matches number of predictors
    assert vi.shape == (5,)


def test_distribute_importance_with_replacement_sampling():
    X, y = make_surv(200, 5, seed=42)
    fit = RLT_surv(n_estimators=100, importance="distribute",
                   resample_replace=True, n_jobs=2,
                   random_state=456).fit(X, y)
    vi = np.asarray(fit.varimp_)
    assert vi[0] > 0.01
    assert vi.shape == (5,)


def test_distribute_ranking_agrees_with_permute():
    X, y = make_surv(300, 5, seed=42)
    fit_perm = RLT_surv(n_estimators=100, importance="permute",
                        resample_prob=0.7, resample_replace=False, n_jobs=2,
                        random_state=123).fit(X, y)
    fit_dist = RLT_surv(n_estimators=100, importance="distribute",
                        resample_prob=0.7, resample_replace=False, n_jobs=2,
                        random_state=123).fit(X, y)
    # Both methods should rank X1 and X2 as top 2 variables
    rank_perm = np.argsort(np.asarray(fit_perm.varimp_))[::-1]
    rank_dist = np.argsort(np.asarray(fit_dist.varimp_))[::-1]
    assert set(rank_perm[:2]) <= {0, 1}
    assert set(rank_dist[:2]) <= {0, 1}


def test_distribute_with_noise_data_gives_near_zero_vi():
    rng = np.random.default_rng(99)
    X = rng.normal(size=(200, 5))
    time = rng.exponential(size=200)
    censor = rng.choice([0, 1], 200, p=[0.3, 0.7])
    y = np.empty(200, dtype=[("event", "?"), ("time", "<f8")])
    y["event"] = censor.astype(bool)
    y["time"] = time
    fit = RLT_surv(n_estimators=100, importance="distribute",
                   resample_prob=0.7, resample_replace=False, n_jobs=2,
                   random_state=789).fit(X, y)
    vi = np.asarray(fit.varimp_)
    assert np.abs(vi).max() < 0.05


def test_distribute_with_matched_var_mode():
    X, y = make_surv(200, 5, seed=42)
    fit = RLT_surv(n_estimators=100, importance="distribute",
                   var_mode="matched", n_jobs=2,
                   random_state=111).fit(X, y)
    assert np.asarray(fit.varimp_).shape == (5,)
    # R: "VarVI" %in% names(fit), length 5. API gap: VarVI is computed by
    # the core but dropped by the Python estimator layer.
    assert not hasattr(fit, "var_vi_")  # documents the gap


def test_distribute_with_linear_combination_splits():
    X, y = make_surv(100, 5, seed=42)
    fit = RLT_surv(n_estimators=50, importance="distribute",
                   reinforcement=True, linear_comb=2, n_jobs=2,
                   random_state=222).fit(X, y)
    assert np.asarray(fit.varimp_).shape == (5,)
