"""Port of RLT tests/testthat/test-survival-basic.R."""

import numpy as np
import pytest

from rlt import RLT_surv
from helpers import generate_survival_data


def test_survival_model_fits_and_returns_correct_class():
    # expect_s3_class(fit, "RLT") -> isinstance + fitted attributes
    d = generate_survival_data(n=80, p=10, seed=1)
    fit = RLT_surv(n_estimators=30, n_jobs=2, verbose=0).fit(d["X"], d["y"])
    assert isinstance(fit, RLT_surv)
    assert hasattr(fit, "forest_")
    assert hasattr(fit, "timepoints_")


def test_survival_predict_returns_survival_matrix():
    d = generate_survival_data(n=80, p=10, seed=1)
    fit = RLT_surv(n_estimators=30, n_jobs=2, verbose=0).fit(d["X"], d["y"])
    pred = fit.predict_survival_function(d["X"])
    assert isinstance(pred, np.ndarray)
    assert pred.ndim == 2
    assert pred.shape[0] == d["n"]


def test_survival_oob_prediction_is_a_matrix():
    d = generate_survival_data(n=80, p=10, seed=1)
    fit = RLT_surv(n_estimators=30, n_jobs=2, verbose=0).fit(d["X"], d["y"])
    # R: fit$Prediction -> OOB hazard/survival matrix
    assert np.asarray(fit.oob_hazard_).ndim == 2
    assert np.asarray(fit.oob_hazard_).shape[0] == d["n"]


def test_survival_prediction_columns_are_time_grid_points():
    d = generate_survival_data(n=80, p=10, seed=1)
    fit = RLT_surv(n_estimators=30, n_jobs=2, verbose=0).fit(d["X"], d["y"])
    assert fit.oob_hazard_.shape[1] > 1
    assert fit.oob_hazard_.shape[1] == len(fit.timepoints_)


def test_survival_predictions_are_non_negative():
    d = generate_survival_data(n=80, p=10, seed=1)
    fit = RLT_surv(n_estimators=30, n_jobs=2, verbose=0).fit(d["X"], d["y"])
    pred = fit.predict_survival_function(d["X"])
    assert np.all(pred >= 0)


def test_survival_predict_survival_curve_is_monotonically_decreasing():
    d = generate_survival_data(n=80, p=10, seed=1)
    fit = RLT_surv(n_estimators=30, n_jobs=2, verbose=0).fit(d["X"], d["y"])
    pred = fit.predict_survival_function(d["X"])
    for i in range(min(5, d["n"])):
        diffs = np.diff(pred[i])
        assert np.all(diffs <= 1e-10)


def test_survival_oob_error_is_finite():
    d = generate_survival_data(n=80, p=10, seed=1)
    fit = RLT_surv(n_estimators=30, n_jobs=2, verbose=0).fit(d["X"], d["y"])
    assert np.isfinite(fit.oob_error_)


def test_logrank_split_rule_works():
    d = generate_survival_data(n=80, p=10, seed=1)
    # expect_error(..., NA) -> no exception raised
    RLT_surv(n_estimators=30, split_rule="logrank", n_jobs=2, verbose=0).fit(
        d["X"], d["y"]
    )


def test_suplogrank_split_rule_works():
    d = generate_survival_data(n=80, p=10, seed=1)
    RLT_surv(n_estimators=30, split_rule="suplogrank", n_jobs=2, verbose=0).fit(
        d["X"], d["y"]
    )


def test_coxgrad_split_rule_works():
    d = generate_survival_data(n=80, p=10, seed=1)
    RLT_surv(n_estimators=30, split_rule="coxgrad", n_jobs=2, verbose=0).fit(
        d["X"], d["y"]
    )


def test_nsplit_0_best_split_works():
    d = generate_survival_data(n=80, p=10, seed=1)
    RLT_surv(n_estimators=30, nsplit=0, n_jobs=2, verbose=0).fit(d["X"], d["y"])


def test_nsplit_positive_random_splits_work():
    d = generate_survival_data(n=80, p=10, seed=1)
    RLT_surv(n_estimators=30, nsplit=3, n_jobs=2, verbose=0).fit(d["X"], d["y"])


def test_same_seed_gives_identical_results():
    d = generate_survival_data(n=80, p=10, seed=1)
    fit1 = RLT_surv(n_estimators=30, random_state=42, n_jobs=1, verbose=0).fit(
        d["X"], d["y"]
    )
    fit2 = RLT_surv(n_estimators=30, random_state=42, n_jobs=1, verbose=0).fit(
        d["X"], d["y"]
    )
    np.testing.assert_array_equal(fit1.oob_hazard_, fit2.oob_hazard_)


def test_importance_true_returns_correct_length_varimp():
    d = generate_survival_data(n=80, p=10, seed=1)
    fit = RLT_surv(n_estimators=30, importance="permute", n_jobs=2, verbose=0).fit(
        d["X"], d["y"]
    )
    assert fit.feature_importances_.shape == (d["p"],)


def test_importance_false_returns_no_varimp():
    # R: expect_null(fit$VarImp); Python raises NotFittedError on access
    from sklearn.exceptions import NotFittedError

    d = generate_survival_data(n=80, p=10, seed=1)
    fit = RLT_surv(n_estimators=30, importance="none", n_jobs=2, verbose=0).fit(
        d["X"], d["y"]
    )
    with pytest.raises(NotFittedError):
        _ = fit.feature_importances_


# TODO: Python API has no var.prob parameter (R: var.prob)
@pytest.mark.xfail(reason="var.prob not available in Python API", strict=True)
def test_var_prob_is_accepted_without_error():
    d = generate_survival_data(n=80, p=10, seed=1)
    vp = np.full(d["p"], 1.0 / d["p"])
    RLT_surv(n_estimators=30, var_prob=vp, n_jobs=2, verbose=0).fit(d["X"], d["y"])


# TODO: Python API has no obs.w parameter (R: obs.w)
@pytest.mark.xfail(reason="obs.w not available in Python API", strict=True)
def test_obs_w_is_accepted_without_error():
    d = generate_survival_data(n=80, p=10, seed=1)
    w = np.random.default_rng(1).random(d["n"])
    RLT_surv(n_estimators=30, obs_weight=w, n_jobs=2, verbose=0).fit(d["X"], d["y"])


# TODO: Python API does not validate NaN in survival times (R errors); the fit
# silently accepts NaN y values.
@pytest.mark.xfail(reason="NaN in y not rejected by Python API", strict=True)
def test_na_in_y_produces_error():
    d = generate_survival_data(n=80, p=10, seed=1)
    d["y"]["time"][4] = np.nan
    with pytest.raises(ValueError):
        RLT_surv(n_estimators=30, n_jobs=2, verbose=0).fit(d["X"], d["y"])


# TODO: R tests NA in the censoring indicator; numpy's boolean ("?") dtype
# cannot represent NA, so this scenario cannot be constructed in the Python API.
@pytest.mark.skip(reason="NA in censor not representable in bool structured dtype")
def test_na_in_censor_produces_error():
    d = generate_survival_data(n=80, p=10, seed=1)
    censor_na = d["censor"].astype(float)
    censor_na[4] = np.nan
    y = np.empty(d["n"], dtype=[("event", "?"), ("time", "<f8")])
    y["event"] = censor_na.astype(bool)
    y["time"] = d["y"]["time"]
    with pytest.raises(ValueError):
        RLT_surv(n_estimators=30, n_jobs=2, verbose=0).fit(d["X"], y)


def test_mismatched_dimensions_produce_error():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(80, 5))
    y = np.empty(50, dtype=[("event", "?"), ("time", "<f8")])
    y["event"] = True
    y["time"] = rng.exponential(size=50)
    with pytest.raises(ValueError):
        RLT_surv(n_estimators=30, n_jobs=2, verbose=0).fit(X, y)


# TODO: Python API coerces arbitrary integer event codes (e.g. 2) to bool
# instead of rejecting invalid censor values as R does.
@pytest.mark.xfail(reason="invalid censor values not rejected by Python API", strict=True)
def test_invalid_censor_values_produce_error():
    d = generate_survival_data(n=80, p=10, seed=1)
    censor_bad = np.random.default_rng(7).choice([0, 1, 2], size=d["n"])
    y = np.empty(d["n"], dtype=[("event", "?"), ("time", "<f8")])
    y["event"] = censor_bad
    y["time"] = d["y"]["time"]
    with pytest.raises(ValueError):
        RLT_surv(n_estimators=30, n_jobs=2, verbose=0).fit(d["X"], y)


def test_subsample_resample_replace_false_works_for_survival():
    d = generate_survival_data(n=80, p=10, seed=1)
    RLT_surv(
        n_estimators=30, resample_replace=False, n_jobs=2, verbose=0
    ).fit(d["X"], d["y"])


def test_survival_model_has_timepoints():
    d = generate_survival_data(n=80, p=10, seed=1)
    fit = RLT_surv(n_estimators=30, n_jobs=2, verbose=0).fit(d["X"], d["y"])
    # R: expect_false(is.null(fit$timepoints))
    assert fit.timepoints_ is not None
    assert len(fit.timepoints_) > 0
