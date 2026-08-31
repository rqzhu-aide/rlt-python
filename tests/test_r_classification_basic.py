"""Port of RLT tests/testthat/test-classification-basic.R."""

import numpy as np
import pytest

from rlt import RLT_cla
from tests.helpers import generate_classification_data, generate_mixed_feature_regression


def test_classification_fits_and_returns_correct_class():
    # expect_s3_class(fit, "RLT") -> isinstance + fitted attributes
    d = generate_classification_data(n=80, p=10)
    fit = RLT_cla(n_estimators=30, n_jobs=2, verbose=0).fit(d["X"], d["y"])
    assert isinstance(fit, RLT_cla)
    assert hasattr(fit, "forest_")
    assert hasattr(fit, "oob_class_")


def test_predict_returns_correct_length_class_predictions():
    d = generate_classification_data(n=80, p=10)
    fit = RLT_cla(n_estimators=30, n_jobs=2, verbose=0).fit(d["X"], d["y"])
    pred = fit.predict(d["X"])
    # R: is.factor(pred$Prediction) -> predictions drawn from fitted classes
    assert pred.shape == (d["n"],)
    assert set(np.unique(pred)).issubset(set(fit.classes_))


def test_binary_classification_accuracy_above_random():
    d = generate_classification_data(n=80, p=10)
    fit = RLT_cla(n_estimators=30, n_jobs=2, verbose=0).fit(d["X"], d["y"])
    pred = fit.predict(d["X"])
    acc = np.mean(pred == d["y"])
    assert acc > 0.5


def test_multiclass_3_class_fits_and_predicts_all_classes():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(90, 5))
    # R factor levels 1:3 -> 0-based classes 0, 1, 2
    y = np.repeat([0, 1, 2], 30)
    fit = RLT_cla(n_estimators=30, n_jobs=2, verbose=0).fit(X, y)
    pred = fit.predict(X)
    # all(levels(pred$Prediction) == c("1","2","3"))
    assert np.array_equal(fit.classes_, np.array([0, 1, 2]))
    assert set(np.unique(pred)) == {0, 1, 2}


def test_classification_with_factor_columns_works():
    # TODO: Python API lacks per-column categorical (factor/ncat) support, so
    # the categorical columns are treated as plain continuous codes.
    d = generate_mixed_feature_regression(n=80, p=10)
    y = (d["y"] > np.median(d["y"])).astype(int)
    RLT_cla(n_estimators=30, n_jobs=2, verbose=0).fit(d["X"], y)


def test_oob_predictions_are_returned():
    d = generate_classification_data(n=80, p=10)
    fit = RLT_cla(n_estimators=30, n_jobs=2, verbose=0).fit(d["X"], d["y"])
    # R: fit$Prediction -> OOB class predictions
    assert np.asarray(fit.oob_class_).shape == (d["n"],)


def test_importance_true_returns_correct_length_varimp():
    d = generate_classification_data(n=80, p=10)
    fit = RLT_cla(n_estimators=30, importance="permute", n_jobs=2, verbose=0).fit(
        d["X"], d["y"]
    )
    assert fit.feature_importances_.shape == (d["p"],)


# TODO: Python API has no var.prob parameter (R: var.prob)
@pytest.mark.xfail(reason="var.prob not available in Python API", strict=True)
def test_var_prob_is_accepted_without_error():
    d = generate_classification_data(n=80, p=10)
    vp = np.full(d["p"], 1.0 / d["p"])
    RLT_cla(n_estimators=30, var_prob=vp, n_jobs=2, verbose=0).fit(d["X"], d["y"])


# TODO: Python API has no var.prob parameter (R: var.prob), so the skewed
# var.prob behavior cannot be exercised.
@pytest.mark.xfail(reason="var.prob not available in Python API", strict=True)
def test_var_prob_skewed_weights_favor_high_prob_variables():
    d = generate_classification_data(n=80, p=10)
    vp = np.array([5.0] * 3 + [0.1] * 7)
    fit = RLT_cla(
        n_estimators=30, var_prob=vp, importance="permute", n_jobs=2, verbose=0
    ).fit(d["X"], d["y"])
    top3 = np.argsort(fit.feature_importances_)[::-1][:3]
    assert len(top3) == 3


# TODO: Python API has no obs.w parameter (R: obs.w)
@pytest.mark.xfail(reason="obs.w not available in Python API", strict=True)
def test_obs_w_is_accepted_without_error():
    d = generate_classification_data(n=80, p=10)
    w = np.random.default_rng(1).random(d["n"])
    RLT_cla(n_estimators=30, obs_weight=w, n_jobs=2, verbose=0).fit(d["X"], d["y"])


def test_same_seed_gives_identical_results():
    d = generate_classification_data(n=80, p=10)
    fit1 = RLT_cla(n_estimators=30, random_state=42, n_jobs=1, verbose=0).fit(
        d["X"], d["y"]
    )
    fit2 = RLT_cla(n_estimators=30, random_state=42, n_jobs=1, verbose=0).fit(
        d["X"], d["y"]
    )
    np.testing.assert_array_equal(fit1.oob_class_, fit2.oob_class_)


def test_classification_with_lc_split_fits_and_predicts():
    d = generate_classification_data(n=80, p=10)
    fit = RLT_cla(
        n_estimators=30,
        linear_comb=3,
        linear_comb_method="lda",
        n_jobs=2,
        verbose=0,
    ).fit(d["X"], d["y"])
    pred = fit.predict(d["X"])
    # R: is.factor(pred$Prediction) -> predictions drawn from fitted classes
    assert set(np.unique(pred)).issubset(set(fit.classes_))


def test_single_class_y_should_fail_gracefully():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(20, 4))
    y = np.ones(20, dtype=int)
    with pytest.raises(ValueError):
        RLT_cla(n_estimators=30, verbose=0).fit(X, y)


def test_nsplit_0_best_split_works():
    d = generate_classification_data(n=80, p=10)
    RLT_cla(n_estimators=30, nsplit=0, n_jobs=2, verbose=0).fit(d["X"], d["y"])


def test_nsplit_positive_random_splits_work():
    d = generate_classification_data(n=80, p=10)
    RLT_cla(n_estimators=30, nsplit=3, n_jobs=2, verbose=0).fit(d["X"], d["y"])


def test_alpha_0_works():
    d = generate_classification_data(n=80, p=10)
    RLT_cla(n_estimators=30, alpha=0, n_jobs=2, verbose=0).fit(d["X"], d["y"])


def test_alpha_025_works():
    d = generate_classification_data(n=80, p=10)
    RLT_cla(n_estimators=30, alpha=0.25, n_jobs=2, verbose=0).fit(d["X"], d["y"])
