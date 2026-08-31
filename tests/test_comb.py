"""M4 tests: linear-combination (reinforcement) forests."""
import numpy as np
import pytest
from sklearn.base import clone

from rlt import RLTRegressor, RLTClassifier, RLTSurvivalForest
from helpers import make_regression, make_classification, make_survival


def test_reg_comb_fit_predict():
    X, y = make_regression(n=200, p=6, seed=1)
    m = RLTRegressor(n_estimators=30, linear_comb=2, nsplit=2,
                     random_state=3).fit(X, y)
    pred = m.predict(X)
    assert pred.shape == (200,)
    assert np.corrcoef(pred, y)[0, 1] > 0.85
    # SplitLoad present and used
    assert "SplitLoad" in m.forest_
    sv = np.asarray(m.forest_["SplitVar"][0])
    sl = np.asarray(m.forest_["SplitLoad"][0])
    assert sv.ndim == 2 and sl.shape == sv.shape
    assert (np.abs(sl).sum(axis=1) > 0).any()


def test_reg_comb_methods():
    X, y = make_regression(n=150, p=5, seed=2)
    for method in ("naive", "lm", "pca", "sir"):
        m = RLTRegressor(n_estimators=10, linear_comb=2,
                         linear_comb_method=method, nsplit=2,
                         random_state=5).fit(X, y)
        assert m.predict(X).shape == (150,)


def test_reg_reinforcement_only():
    X, y = make_regression(n=200, p=6, seed=3)
    m = RLTRegressor(n_estimators=30, reinforcement=True, nsplit=2,
                     random_state=7).fit(X, y)
    assert np.corrcoef(m.predict(X), y)[0, 1] > 0.85
    assert "SplitLoad" not in m.forest_  # Uni forest with embed splits


def test_cla_comb_fit_predict():
    X, y = make_classification(n=200, p=6, seed=4)
    m = RLTClassifier(n_estimators=30, linear_comb=2, nsplit=2,
                      random_state=3).fit(X, y)
    assert m.predict(X).shape == (200,)
    proba = m.predict_proba(X)
    assert proba.shape == (200, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert np.mean(m.predict(X) == y) > 0.8


def test_surv_comb_fit_predict():
    X, ys = make_survival(n=200, p=6, seed=5)
    time, event = ys["time"], ys["event"].astype(int)
    m = RLTSurvivalForest(n_estimators=30, linear_comb=2, nsplit=2,
                          random_state=3).fit(X, (time, event))
    S = m.predict(X)
    assert S.shape[0] == 200
    assert np.all(np.diff(S, axis=1) <= 1e-12)  # survival is decreasing
    assert m.score(X, ys) > 0.65


def test_comb_sklearn_clone():
    X, y = make_regression(n=100, p=4, seed=6)
    m = RLTRegressor(n_estimators=5, linear_comb=2, random_state=1)
    m2 = clone(m).fit(X, y)
    assert np.allclose(m2.predict(X), m.fit(X, y).predict(X))


def test_embed_protect_default_nonzero():
    # embed_protect=None must resolve to ceil(log(n)) >= 1 (a 0 would
    # underflow the core's var-protect subvec)
    X, y = make_regression(n=120, p=5, seed=7)
    m = RLTRegressor(n_estimators=5, linear_comb=2, random_state=2).fit(X, y)
    assert m.params_.embed_protect >= 1
