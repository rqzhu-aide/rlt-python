"""Port of RLT testthat/test-regression-lc.R (regression linear combination).

Param map: ntrees->n_estimators, linear.comb->linear_comb,
linear.comb.method->linear_comb_method, ncores->n_jobs, var.prob->var_prob,
seed->random_state.
"""
import numpy as np
import pytest

from rlt import RLT_reg
from helpers import generate_simple_regression, generate_mixed_feature_regression


def _fit(X, y, **kw):
    kw.setdefault("n_jobs", 2)
    kw.setdefault("random_state", 1)
    return RLT_reg(**kw).fit(X, y)


@pytest.mark.parametrize("method", ["naive", "lm", "pca", "sir"])
def test_lc_method_works(method):
    # R: "naive method (1) works" ... "sir method (4) works"
    d = generate_simple_regression(n=80, p=10)
    _fit(d["X"], d["y"], n_estimators=30, linear_comb=3,
         linear_comb_method=method)


@pytest.mark.parametrize("m", [1, 2, 3, 4])
def test_integer_method_codes_work(m):
    # R: integer codes 1-4 accepted without error. Ported: the Python layer
    # now accepts integer codes directly (no stringification/fallback).
    d = generate_simple_regression(n=80, p=10)
    fit = _fit(d["X"], d["y"], n_estimators=20, linear_comb=3,
               linear_comb_method=m)
    assert fit.params_.linear_comb_method == m


def test_lc_with_categorical_predictors():
    # R uses a data.frame with factor columns. The Python API has no
    # per-column categorical (ncat) metadata, so integer level codes are
    # passed as numeric columns -- same shapes, no factor semantics.
    d = generate_mixed_feature_regression(n=80, p=10)
    _fit(d["X"], d["y"], n_estimators=30, linear_comb=3,
         linear_comb_method="naive")


def test_lc_predictions_comparable_to_non_lc():
    d = generate_simple_regression(n=80, p=10)
    fit_lc = _fit(d["X"], d["y"], n_estimators=30, linear_comb=3,
                  linear_comb_method="naive")
    fit_base = _fit(d["X"], d["y"], n_estimators=30)
    mse_lc = np.mean((d["y"] - fit_lc.predict(d["X"])) ** 2)
    mse_base = np.mean((d["y"] - fit_base.predict(d["X"])) ** 2)
    # LC should not be catastrophically worse
    assert mse_lc < 10 * mse_base


def test_small_sample_lc_fits():
    d = generate_simple_regression(n=30, p=5)
    _fit(d["X"], d["y"], n_estimators=20, linear_comb=3,
         linear_comb_method="sir")


def test_lc_sir_beats_mean_baseline():
    d = generate_simple_regression(n=80, p=10)
    fit = _fit(d["X"], d["y"], n_estimators=30, linear_comb=3,
               linear_comb_method="sir")
    pred = fit.predict(d["X"])
    mse_model = np.mean((d["y"] - pred) ** 2)
    mse_mean = np.mean((d["y"] - d["y"].mean()) ** 2)
    assert mse_model < mse_mean


def test_invalid_linear_comb_method_triggers_warning():
    # R: expect_warning + reset to code 1 (naive). Ported: the Python layer
    # warns and resets linear_comb_method to 1, mirroring RegForest.r.
    d = generate_simple_regression(n=80, p=10)
    with pytest.warns(UserWarning, match="not recognized"):
        fit = _fit(d["X"], d["y"], n_estimators=20, linear_comb=3,
                   linear_comb_method="bogus")
    assert fit.params_.linear_comb_method == 1


def test_var_prob_accepted_alongside_lc():
    d = generate_simple_regression(n=80, p=10)
    vp = np.full(d["p"], 1 / d["p"])
    RLT_reg(n_estimators=30, linear_comb=3, linear_comb_method="naive",
            n_jobs=2).fit(d["X"], d["y"], var_prob=vp)


def test_lc_works_with_p_larger_than_n():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(80, 50))
    y = X[:, 0] + rng.normal(size=80)
    _fit(X, y, n_estimators=20, linear_comb=3, linear_comb_method="naive")
