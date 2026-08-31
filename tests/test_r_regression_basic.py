"""Port of RLT tests/testthat/test-regression-basic.R."""

import numpy as np
import pytest

from rlt import RLT_reg
from helpers import generate_simple_regression, generate_mixed_feature_regression


def test_regression_fits_and_returns_correct_class():
    # expect_s3_class(fit, "RLT") -> isinstance + fitted attributes
    d = generate_simple_regression(n=80, p=10)
    fit = RLT_reg(n_estimators=30, n_jobs=2, verbose=0).fit(d["X"], d["y"])
    assert isinstance(fit, RLT_reg)
    assert hasattr(fit, "forest_")
    assert hasattr(fit, "oob_prediction_")


def test_predict_returns_correct_length_numeric_predictions():
    d = generate_simple_regression(n=80, p=10)
    fit = RLT_reg(n_estimators=30, n_jobs=2, verbose=0).fit(d["X"], d["y"])
    pred = fit.predict(d["X"])
    assert pred.dtype == np.float64
    assert pred.shape == (d["n"],)


def test_oob_predictions_returned_with_correct_length():
    d = generate_simple_regression(n=80, p=10)
    fit = RLT_reg(n_estimators=30, n_jobs=2, verbose=0).fit(d["X"], d["y"])
    # R: fit$Prediction (training/OOB prediction)
    assert np.asarray(fit.oob_prediction_).shape == (d["n"],)


def test_training_predictions_correlate_with_true_values():
    d = generate_simple_regression(n=80, p=10)
    fit = RLT_reg(n_estimators=30, n_jobs=2, verbose=0).fit(d["X"], d["y"])
    cor_val = np.corrcoef(fit.oob_prediction_, d["y"])[0, 1]
    assert cor_val > 0


def test_importance_true_returns_correct_length_varimp():
    d = generate_simple_regression(n=80, p=10)
    fit = RLT_reg(n_estimators=30, importance="permute", n_jobs=2, verbose=0).fit(
        d["X"], d["y"]
    )
    assert fit.feature_importances_.shape == (d["p"],)


def test_importance_false_returns_no_varimp():
    # R: expect_null(fit$VarImp); Python raises NotFittedError on access
    from sklearn.exceptions import NotFittedError

    d = generate_simple_regression(n=80, p=10)
    fit = RLT_reg(n_estimators=30, importance="none", n_jobs=2, verbose=0).fit(
        d["X"], d["y"]
    )
    with pytest.raises(NotFittedError):
        _ = fit.feature_importances_


def test_importance_identifies_signal_variables():
    d = generate_simple_regression(n=80, p=10)
    fit = RLT_reg(n_estimators=30, importance="permute", n_jobs=2, verbose=0).fit(
        d["X"], d["y"]
    )
    top2 = np.argsort(fit.feature_importances_)[::-1][:2]
    # X1 and X2 are signal (0-based columns 0 and 1); at least 1 in top 2
    assert np.isin([0, 1], top2).sum() >= 1


# TODO: Python API has no var.prob parameter (R: var.prob)
@pytest.mark.xfail(reason="var.prob not available in Python API", strict=True)
def test_var_prob_is_accepted_without_error():
    d = generate_simple_regression(n=80, p=10)
    vp = np.full(d["p"], 1.0 / d["p"])
    RLT_reg(n_estimators=30, var_prob=vp, n_jobs=2, verbose=0).fit(d["X"], d["y"])


# TODO: Python API has no obs.w parameter (R: obs.w)
@pytest.mark.xfail(reason="obs.w not available in Python API", strict=True)
def test_obs_w_is_accepted_without_error():
    d = generate_simple_regression(n=80, p=10)
    w = np.random.default_rng(1).random(d["n"])
    RLT_reg(n_estimators=30, obs_weight=w, n_jobs=2, verbose=0).fit(d["X"], d["y"])


def test_same_seed_gives_identical_results():
    d = generate_simple_regression(n=80, p=10)
    fit1 = RLT_reg(n_estimators=30, random_state=42, n_jobs=1, verbose=0).fit(
        d["X"], d["y"]
    )
    fit2 = RLT_reg(n_estimators=30, random_state=42, n_jobs=1, verbose=0).fit(
        d["X"], d["y"]
    )
    np.testing.assert_array_equal(fit1.oob_prediction_, fit2.oob_prediction_)


def test_different_seeds_give_different_results():
    d = generate_simple_regression(n=80, p=10)
    fit1 = RLT_reg(n_estimators=30, random_state=1, n_jobs=1, verbose=0).fit(
        d["X"], d["y"]
    )
    fit2 = RLT_reg(n_estimators=30, random_state=2, n_jobs=1, verbose=0).fit(
        d["X"], d["y"]
    )
    assert not np.array_equal(fit1.oob_prediction_, fit2.oob_prediction_)


# TODO: Python API lacks per-column categorical (factor/ncat) support, so the
# categorical columns are treated as plain continuous codes rather than as
# factors as in R.  The fit itself must still succeed without error.
def test_factor_columns_in_x_are_handled():
    d = generate_mixed_feature_regression(n=80, p=10)
    RLT_reg(n_estimators=30, n_jobs=2, verbose=0).fit(d["X"], d["y"])


def test_nsplit_0_best_split_works():
    d = generate_simple_regression(n=80, p=10)
    # expect_error(..., NA) -> no exception raised
    RLT_reg(n_estimators=30, nsplit=0, n_jobs=2, verbose=0).fit(d["X"], d["y"])


def test_nsplit_positive_random_splits_work():
    d = generate_simple_regression(n=80, p=10)
    RLT_reg(n_estimators=30, nsplit=3, n_jobs=2, verbose=0).fit(d["X"], d["y"])


def test_alpha_0_works():
    d = generate_simple_regression(n=80, p=10)
    RLT_reg(n_estimators=30, alpha=0, n_jobs=2, verbose=0).fit(d["X"], d["y"])


def test_alpha_025_works():
    d = generate_simple_regression(n=80, p=10)
    RLT_reg(n_estimators=30, alpha=0.25, n_jobs=2, verbose=0).fit(d["X"], d["y"])


def test_oob_error_is_finite():
    d = generate_simple_regression(n=80, p=10)
    fit = RLT_reg(n_estimators=30, n_jobs=2, verbose=0).fit(d["X"], d["y"])
    assert np.isfinite(fit.oob_error_)


def test_train_test_split_test_predictions_correlate_with_truth():
    d = generate_simple_regression(n=100, p=10)
    train_idx = np.arange(70)
    test_idx = np.arange(70, 100)
    fit = RLT_reg(n_estimators=30, n_jobs=2, verbose=0).fit(
        d["X"][train_idx], d["y"][train_idx]
    )
    pred = fit.predict(d["X"][test_idx])
    assert np.corrcoef(pred, d["y"][test_idx])[0, 1] > 0.3


def test_importance_distribute_returns_correct_length_varimp():
    d = generate_simple_regression(n=80, p=10)
    fit = RLT_reg(n_estimators=30, importance="distribute", n_jobs=2, verbose=0).fit(
        d["X"], d["y"]
    )
    assert fit.feature_importances_.shape == (d["p"],)


def test_resample_parameters_can_be_set_manually():
    d = generate_simple_regression(n=80, p=10)
    RLT_reg(
        n_estimators=30,
        resample_replace=False,
        resample_prob=0.632,
        n_jobs=2,
        verbose=0,
    ).fit(d["X"], d["y"])
