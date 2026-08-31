"""OOB self-kernel tests (port of the RLT >= 6.1.0 oob = TRUE feature)."""
import numpy as np
import pytest

from rlt import RLT_cla, RLT_reg
from helpers import make_classification, make_regression


def _fit_reg(n=100, p=4, seed=5, **kw):
    X, y = make_regression(n=n, p=p, seed=seed)
    m = RLT_reg(n_estimators=40, resample_track=True, nsplit=2,
                random_state=5, **kw).fit(X, y)
    return X, y, m


def test_oob_kernel_returns_dict_with_KNC():
    X, y, m = _fit_reg()
    out = m.forest_kernel(X, oob=True)
    assert set(out.keys()) == {"Kernel", "N", "C"}
    K, N, C = out["Kernel"], out["N"], out["C"]
    assert K.shape == N.shape == C.shape == (100, 100)
    assert K.dtype.kind == "f" and N.dtype.kind == "i" and C.dtype.kind == "i"
    assert (K >= 0).all() and (K <= 1).all()
    assert (C <= N).all()
    # symmetric counts
    assert (N == N.T).all() and (C == C.T).all() and np.allclose(K, K.T)


def test_oob_kernel_diag_and_kernel_equals_c_over_n():
    X, y, m = _fit_reg()
    K, N, C = (m.forest_kernel(X, oob=True)[k] for k in ("Kernel", "N", "C"))
    # diagonal: an OOB observation shares its own leaf, so C_ii == N_ii
    assert (np.diag(C) == np.diag(N)).all()
    mask = np.diag(N) > 0
    assert (np.diag(K)[mask] == 1.0).all()
    # off-diagonal pairs with zero co-occurrence get Kernel = 0
    assert (K[N == 0] == 0).all()
    # Kernel = C / N where N > 0
    nz = N > 0
    assert np.allclose(K[nz], C[nz] / N[nz])


def test_oob_kernel_consistent_with_full_self_kernel():
    # where both obs are OOB in all trees (impossible here), Kernel would
    # match the plain self kernel; instead check a computable identity:
    # C_ij <= plain self-kernel counts (same trees, restricted to OOB)
    X, y, m = _fit_reg()
    Kfull = m.forest_kernel(X)
    out = m.forest_kernel(X, oob=True)
    assert (out["C"] <= Kfull).all()


def test_oob_kernel_counts_respect_obstrack():
    # N_ij must equal the number of trees where both i and j are OOB
    X, y, m = _fit_reg()
    N = m.forest_kernel(X, oob=True)["N"]
    oob = (m.obstrack_ == 0)  # (n, ntrees) boolean
    for i, j in ((0, 0), (0, 1), (3, 17), (25, 71), (42, 99)):
        expected = int((oob[i] & oob[j]).sum())
        assert N[i, j] == expected


def test_oob_kernel_comb_forest():
    X, y = make_regression(n=80, p=4, seed=13)
    m = RLT_reg(n_estimators=20, linear_comb=2, resample_track=True,
                nsplit=2, random_state=5).fit(X, y)
    out = m.forest_kernel(X, oob=True)
    K, N, C = out["Kernel"], out["N"], out["C"]
    assert K.shape == (80, 80)
    assert (C <= N).all()
    assert np.allclose(K[N > 0], (C / np.maximum(N, 1))[N > 0])
    assert (np.diag(C) == np.diag(N)).all()


def test_oob_kernel_classification():
    X, y = make_classification(n=100, p=4, seed=6)
    m = RLT_cla(n_estimators=40, resample_track=True, nsplit=2,
                random_state=5).fit(X, y)
    out = m.forest_kernel(X, oob=True)
    assert (out["C"] <= out["N"]).all()
    assert (out["Kernel"] >= 0).all() and (out["Kernel"] <= 1).all()


def test_oob_kernel_requires_resample_track():
    X, y = make_regression(n=60, p=3, seed=14)
    m = RLT_reg(n_estimators=5, random_state=1).fit(X, y)
    with pytest.raises(ValueError, match="resample_track"):
        m.forest_kernel(X, oob=True)


def test_oob_kernel_requires_training_data():
    X, y, m = _fit_reg()
    with pytest.raises(ValueError, match="training data"):
        m.forest_kernel(X[:50], oob=True)


def test_oob_kernel_rejects_x2():
    X, y, m = _fit_reg()
    with pytest.raises(ValueError, match="self-kernel"):
        m.forest_kernel(X, X, oob=True)


def test_oob_kernel_kernel_diagonal_is_one_when_oob_exists():
    X, y, m = _fit_reg(n=120)
    out = m.forest_kernel(X, oob=True)
    K, N = out["Kernel"], out["N"]
    assert (N > 0).any()
    assert np.all((np.diag(K)[np.diag(N) > 0] == 1.0))
    # every observation is OOB in at least one of 40 bootstrap trees
    assert (np.diag(N) > 0).all()
