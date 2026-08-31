"""M5 tests: importance modes, kernels, and survival bands."""
import numpy as np

from rlt import RLTRegressor, RLTSurvivalForest, get_surv_band
from helpers import make_regression, make_survival


def test_importance_modes():
    X, y = make_regression(n=250, p=6, seed=11)
    for imp in ("permute", "distribute"):
        m = RLTRegressor(n_estimators=50, importance=imp, nsplit=2,
                         random_state=5).fit(X, y)
        vi = m.feature_importances_
        assert vi.shape == (6,)
        top2 = set(np.argsort(vi)[::-1][:2].tolist())
        assert {0, 1} <= top2  # helpers' DGP: y = x0 + x1^2 + noise
        # tree-averaged VI can carry small negative MC noise (as in R);
        # noise variables must stay negligible vs signal variables
        assert vi[0] > 5 * abs(vi[4]) and vi[1] > 5 * abs(vi[4])


def test_kernel_self_and_cross():
    X, y = make_regression(n=100, p=4, seed=12)
    m = RLTRegressor(n_estimators=40, resample_track=True, nsplit=2,
                     random_state=5).fit(X, y)
    K = m.forest_kernel(X)
    assert K.shape == (100, 100)
    assert (K == K.T).all()
    assert (K >= 0).all()
    Kc = m.forest_kernel(X[:20], X)
    assert Kc.shape == (20, 100)
    Kt = m.forest_kernel(X[:20], X, vs_train=True)
    assert Kt.shape == (20, 100)
    # Kt counts only inbag X2 observations; wherever Kc is zero Kt must be
    # zero too (same tree structure), and OOB obs make Kt sparser than Kc
    assert np.all(Kt[Kc == 0] == 0)


def test_kernel_comb_forest():
    X, y = make_regression(n=80, p=4, seed=13)
    m = RLTRegressor(n_estimators=20, linear_comb=2, nsplit=2,
                     random_state=5).fit(X, y)
    K = m.forest_kernel(X)
    assert K.shape == (80, 80)
    assert K.diagonal()[0] == 20  # self co-occurrence = ntrees


def test_kernel_vs_train_requires_tracking():
    X, y = make_regression(n=60, p=3, seed=14)
    m = RLTRegressor(n_estimators=5, random_state=1).fit(X, y)
    try:
        m.forest_kernel(X[:5], X, vs_train=True)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_surv_band_naive_and_smoothed():
    X, ys = make_survival(n=120, p=4, seed=15)
    m = RLTSurvivalForest(n_estimators=100, var_mode="matched", nsplit=2,
                          random_state=5).fit(X, ys)
    Xt = X[:4]
    S, cov = m.predict_var(Xt)
    assert cov.ndim == 3 and cov.shape[0] == 4

    b1 = get_surv_band(m, Xt, i=0, alpha=0.05, approach="naive", nsim=1000)
    assert b1["lower"].shape == S.shape
    assert np.all(b1["lower"] <= b1["upper"] + 1e-12)
    assert np.all(b1["lower"] > 0)

    b2 = get_surv_band(m, Xt, i=0, alpha=0.05, approach="smoothed",
                       nsim=1000, k_rank=6)
    assert b2["lower"].shape == S.shape
    d = b2["diagnostics"][0]["var_explained"]
    assert 0 < d["var_explained"] <= 1.000001
    assert d["k_rank"] == 6


def test_surv_band_proportion_mode():
    X, ys = make_survival(n=100, p=3, seed=16)
    m = RLTSurvivalForest(n_estimators=60, var_mode="matched", nsplit=2,
                          random_state=5).fit(X, ys)
    b = get_surv_band(m, X[:2], i=0, approach="smoothed", k_mode="proportion",
                      k_prop=0.9, nsim=500, band_grid_size=15)
    assert b["lower"].shape[1] <= 15
    k = b["diagnostics"][0]["eig"]["k_used"]
    assert 1 <= k <= b["lower"].shape[1]
