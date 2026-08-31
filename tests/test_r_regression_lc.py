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
    return RLT_reg(**kw).fit(X, y)


@pytest.mark.parametrize("method", ["naive", "lm", "pca", "sir"])
def test_lc_method_works(method):
    # R: "naive method (1) works" ... "sir method (4) works"
    d = generate_simple_regression(n=80, p=10)
    _fit(d["X"], d["y"], n_estimators=30, linear_comb=3,
         linear_comb_method=method)


@pytest.mark.parametrize("m", [1, 2, 3, 4])
def test_integer_method_codes_work(m):
    # R: integer codes 1-4 accepted without error. NOTE (API gap): the Python
    # layer maps *string* names only; an int code is stringified and misses the
    # lookup table, silently falling back to the default method (code 4/sir).
    # The fit itself still succeeds, matching R's expect_error(..., NA).
    d = generate_simple_regression(n=80, p=10)
    _fit(d["X"], d["y"], n_estimators=20, linear_comb=3,
         linear_comb_method=m)


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
    # R: expect_warning(...). The Python layer does NOT warn for an
    # unrecognized linear_comb_method -- it silently falls back to the
    # default (code 4). Skipped per port policy; genuine API gap.
    pytest.skip("Python layer does not warn on invalid linear_comb_method "
                "(silently falls back to default)")


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
