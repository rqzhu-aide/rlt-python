"""Basic correctness tests: fit + predict for all three model types."""

import numpy as np
import pytest
from sklearn.base import clone

from helpers import make_classification, make_regression, make_survival

from rlt import RLTClassifier, RLTRegressor, RLTSurvivalForest


def test_regressor_basic():
    X, y = make_regression(n=300, seed=1)
    m = RLTRegressor(n_estimators=100, random_state=7).fit(X, y)
    pred = m.predict(X)
    assert pred.shape == (300,)
    # in-sample forest should track the signal reasonably well
    assert np.corrcoef(pred, y)[0, 1] > 0.8
    # OOB error exists
    assert hasattr(m, "oob_error_")
    assert np.isfinite(m.oob_error_)
    # sklearn plumbing
    assert clone(m).get_params()["n_estimators"] == 100
    assert m.n_features_in_ == 5


def test_regressor_reproducible():
    X, y = make_regression(n=150, seed=2)
    a = RLTRegressor(n_estimators=50, random_state=11).fit(X, y).predict(X)
    b = RLTRegressor(n_estimators=50, random_state=11).fit(X, y).predict(X)
    assert np.allclose(a, b)


def test_classifier_basic():
    X, y = make_classification(n=300, seed=3)
    m = RLTClassifier(n_estimators=100, random_state=7).fit(X, y)
    labels = m.predict(X)
    assert labels.shape == (300,)
    assert set(np.unique(labels)).issubset(set(np.unique(y)))
    proba = m.predict_proba(X)
    assert proba.shape == (300, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    acc = np.mean(labels == y)
    assert acc > 0.7
    # classes_ mapping preserved
    m2 = RLTClassifier(n_estimators=30, random_state=1).fit(X, y.astype(str))
    assert set(m2.classes_) == {"0", "1"}
    assert set(m2.predict(X)) <= {"0", "1"}


def test_classifier_feature_importances():
    X, y = make_classification(n=200, seed=4)
    m = RLTClassifier(n_estimators=100, importance="distribute",
                      random_state=5).fit(X, y)
    imp = m.feature_importances_
    assert imp.shape == (5,)
    assert imp[0] >= imp[4]  # informative variables dominate


def test_survival_basic():
    X, y = make_survival(n=300, seed=5)
    m = RLTSurvivalForest(n_estimators=100, random_state=7).fit(X, y)
    S = m.predict(X)
    assert S.shape == (300, m.timepoints_.shape[0])
    assert np.all(S <= 1.0 + 1e-12) and np.all(S > -1e-12)
    # survival curves should be non-increasing in time
    assert np.all(np.diff(S, axis=1) <= 1e-12)
    # c-index beats random
    assert m.score(X, y) > 0.6
    chf = m.predict_cumulative_hazard_function(X)
    assert np.all(np.diff(chf, axis=1) >= -1e-12)


def test_survival_tuple_y():
    X, y = make_survival(n=150, seed=6)
    m1 = RLTSurvivalForest(n_estimators=30, random_state=2).fit(X, y)
    m2 = RLTSurvivalForest(n_estimators=30, random_state=2).fit(
        X, (y["time"], y["event"])
    )
    assert np.allclose(m1.predict(X), m2.predict(X))


def test_input_validation():
    X, y = make_regression(n=60, seed=7)
    X[0, 0] = np.nan
    with pytest.raises(ValueError):
        RLTRegressor(n_estimators=10).fit(X, y)
    with pytest.raises(ValueError):
        RLTRegressor(n_estimators=10).fit(X[:50], y)


def test_var_modes_regression():
    X, y = make_regression(n=200, seed=8)
    m = RLTRegressor(n_estimators=100, resample_replace=False,
                     resample_prob=0.5, var_mode="matched",
                     importance="distribute", random_state=9).fit(X, y)
    pred, var = m.predict_var(X[:20])
    assert var.shape == (20,)
    finite = np.isfinite(var)
    assert finite.any()
    assert np.all(var[finite] >= 0)


def test_var_modes_regression_ij():
    X, y = make_regression(n=200, seed=9)
    m = RLTRegressor(n_estimators=100, var_mode="ij", resample_track=True,
                     random_state=10).fit(X, y)
    pred, var = m.predict_var(X[:20])
    assert var.shape == (20,)
    assert np.isfinite(var).any()
