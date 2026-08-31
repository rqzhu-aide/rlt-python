"""Port of RLT testthat/test-classification-probability.R.

Param map: ntrees->n_estimators, linear.comb->linear_comb,
linear.comb.method->linear_comb_method, importance TRUE->"permute",
ncores->n_jobs, seed->random_state.

R's fit$Prob maps to RLT_cla.oob_prob_ (OOB class probabilities) and
predict()$Prob maps to .predict_proba(X).
"""
import numpy as np
import pytest

from rlt import RLT_cla
from helpers import generate_classification_data


def _fit(X, y, **kw):
    kw.setdefault("n_jobs", 2)
    kw.setdefault("random_state", 1)
    return RLT_cla(**kw).fit(X, y)


@pytest.fixture()
def d():
    return generate_classification_data(n=80, p=10)


# -- binary OOB probabilities ------------------------------------------------

def test_binary_oob_prob_dimensions(d):
    fit = _fit(d["X"], d["y"], n_estimators=30)
    P = np.asarray(fit.oob_prob_)
    assert P.ndim == 2
    assert P.shape == (d["n"], 2)


def test_binary_oob_prob_rows_sum_to_one(d):
    fit = _fit(d["X"], d["y"], n_estimators=30)
    np.testing.assert_allclose(fit.oob_prob_.sum(axis=1), np.ones(d["n"]),
                               atol=1e-12)


def test_binary_oob_prob_values_in_unit_interval(d):
    fit = _fit(d["X"], d["y"], n_estimators=30)
    P = fit.oob_prob_
    assert np.all((P >= 0) & (P <= 1))


def test_binary_oob_prediction_agrees_with_argmax(d):
    fit = _fit(d["X"], d["y"], n_estimators=30)
    expected = fit.classes_[np.argmax(fit.oob_prob_, axis=1)]
    np.testing.assert_array_equal(fit.oob_class_, expected.astype(int))


# -- binary predict() probabilities -------------------------------------------

def test_binary_predict_prob_dimensions(d):
    tr = slice(0, 60)
    fit = _fit(d["X"][tr], d["y"][tr], n_estimators=30)
    P = fit.predict_proba(d["X"][60:])
    assert P.shape == (20, 2)


def test_binary_predict_prob_rows_sum_to_one(d):
    tr = slice(0, 60)
    fit = _fit(d["X"][tr], d["y"][tr], n_estimators=30)
    np.testing.assert_allclose(fit.predict_proba(d["X"][60:]).sum(axis=1),
                               np.ones(20), atol=1e-12)


def test_binary_predict_prob_values_in_unit_interval(d):
    tr = slice(0, 60)
    fit = _fit(d["X"][tr], d["y"][tr], n_estimators=30)
    P = fit.predict_proba(d["X"][60:])
    assert np.all((P >= 0) & (P <= 1))


def test_binary_prediction_matches_argmax_of_prob(d):
    tr = slice(0, 60)
    fit = _fit(d["X"][tr], d["y"][tr], n_estimators=30)
    Xte = d["X"][60:]
    expected = fit.classes_[np.argmax(fit.predict_proba(Xte), axis=1)]
    np.testing.assert_array_equal(fit.predict(Xte), expected)


# -- 3-class ------------------------------------------------------------------

def _three_class(n=90, p=5, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = np.repeat([1, 2, 3], n // 3)
    return X, y


def test_three_class_oob_prob_dimensions():
    X, y = _three_class()
    fit = _fit(X, y, n_estimators=30)
    assert np.asarray(fit.oob_prob_).shape == (90, 3)


def test_three_class_oob_prob_rows_sum_to_one():
    X, y = _three_class()
    fit = _fit(X, y, n_estimators=30)
    np.testing.assert_allclose(fit.oob_prob_.sum(axis=1), np.ones(90),
                               atol=1e-12)


def test_three_class_oob_prob_values_in_unit_interval():
    X, y = _three_class()
    fit = _fit(X, y, n_estimators=30)
    P = fit.oob_prob_
    assert np.all((P >= 0) & (P <= 1))


def test_three_class_oob_prob_is_informative():
    # classes driven by eta = X1 - X2 (thresholds 0.5 / -0.5)
    rng = np.random.default_rng(42)
    X = rng.normal(size=(90, 5))
    eta = X[:, 0] - X[:, 1]
    y = np.where(eta > 0.5, 1, np.where(eta > -0.5, 2, 3))
    fit = _fit(X, y, n_estimators=30)
    assert np.mean(fit.oob_prob_.max(axis=1)) > 1 / 3


def test_three_class_predict_prob_dimensions():
    X, y = _three_class()
    fit = _fit(X[:70], y[:70], n_estimators=30)
    assert fit.predict_proba(X[70:]).shape == (20, 3)


def test_three_class_predict_prob_rows_sum_to_one():
    X, y = _three_class()
    fit = _fit(X[:70], y[:70], n_estimators=30)
    np.testing.assert_allclose(fit.predict_proba(X[70:]).sum(axis=1),
                               np.ones(20), atol=1e-12)


def test_three_class_prediction_matches_argmax_of_prob():
    X, y = _three_class()
    fit = _fit(X[:70], y[:70], n_estimators=30)
    Xte = X[70:]
    expected = fit.classes_[np.argmax(fit.predict_proba(Xte), axis=1)]
    np.testing.assert_array_equal(fit.predict(Xte).astype(int), expected)


# -- 4-class ------------------------------------------------------------------

def test_four_class_oob_prob_has_four_columns():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(80, 5))
    y = np.repeat([1, 2, 3, 4], 20)
    fit = _fit(X, y, n_estimators=30)
    assert np.asarray(fit.oob_prob_).shape[1] == 4


# -- misc ---------------------------------------------------------------------

def test_lc_classification_returns_valid_probabilities(d):
    fit = _fit(d["X"], d["y"], n_estimators=30, linear_comb=3,
               linear_comb_method="lda")
    P = np.asarray(fit.oob_prob_)
    assert P.shape == (d["n"], 2)
    np.testing.assert_allclose(P.sum(axis=1), np.ones(d["n"]), atol=1e-12)


def test_prob_returned_when_importance_enabled(d):
    fit = _fit(d["X"], d["y"], n_estimators=30, importance="permute")
    assert np.asarray(fit.oob_prob_).shape == (d["n"], 2)


def test_well_separated_classes_produce_high_confidence(d):
    rng = np.random.default_rng(42)
    X = np.vstack([rng.normal(loc=3.0, size=(40, 5)),
                   rng.normal(loc=-3.0, size=(40, 5))])
    y = np.repeat(["A", "B"], 40)
    fit = _fit(X, y, n_estimators=30)
    assert np.mean(fit.oob_prob_.max(axis=1)) > 0.9
