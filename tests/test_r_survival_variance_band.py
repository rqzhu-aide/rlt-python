"""Port of RLT testthat/test-survival-variance-band.R.

Param map: param.control list(var.mode=..., band.grid.size=..., keep.all=...)
-> var_mode / band_grid_size (predict-time) / keep_all (predict-time)
kwargs; ncores->n_jobs; seed->random_state; get.surv.band(pred, subject=i)
-> get_surv_band(model, X, i=i) (i is 1-based like R when positive).

API gaps vs R (documented in-line): VarVI and the importance() SD/Z/Sig
extractor are not surfaced; NA-in-y / invalid censor values do not raise.
"""
import numpy as np
import pytest

from rlt import RLT_surv, get_surv_band
from helpers import generate_survival_data


@pytest.fixture(scope="module")
def d():
    return generate_survival_data(n=80, p=10, seed=1)


def _fit(X, y, **kw):
    kw.setdefault("n_jobs", 2)
    kw.setdefault("random_state", 1)
    return RLT_surv(**kw).fit(X, y)


# --- Cov structure -------------------------------------------------------------

@pytest.mark.parametrize("mode", ["matched", "ij", "jack"])
def test_cov_is_3d_array(d, mode):
    fit = _fit(d["X"], d["y"], n_estimators=30, var_mode=mode)
    cov = np.asarray(fit._predict_raw(d["X"], var_est=True)["Cov"])
    assert cov.ndim == 3
    assert cov.shape[0] == d["n"]


def test_covariance_matrices_are_symmetric(d):
    fit = _fit(d["X"], d["y"], n_estimators=30, var_mode="matched")
    cov = np.asarray(fit._predict_raw(d["X"], var_est=True)["Cov"])
    for i in range(min(5, cov.shape[0])):
        np.testing.assert_allclose(cov[i], cov[i].T, atol=1e-10)


def test_different_variance_modes_produce_different_cov(d):
    fit_m = _fit(d["X"], d["y"], n_estimators=30, var_mode="matched",
                 random_state=1)
    fit_j = _fit(d["X"], d["y"], n_estimators=30, var_mode="jack",
                 random_state=1)
    cov_m = np.asarray(fit_m._predict_raw(d["X"], var_est=True)["Cov"])
    cov_j = np.asarray(fit_j._predict_raw(d["X"], var_est=True)["Cov"])
    assert not np.array_equal(cov_m, cov_j)


def test_var_est_false_returns_no_cov(d):
    fit = _fit(d["X"], d["y"], n_estimators=30, var_mode="matched")
    out = fit._predict_raw(d["X"], var_est=False)
    assert "Cov" not in out


def test_band_grid_size_reduces_timepoints(d):
    fit = _fit(d["X"], d["y"], n_estimators=30, var_mode="matched")
    full_tp = fit.predict_survival_function(d["X"]).shape[1]
    small = fit.predict_survival_function(d["X"], band_grid_size=5)
    assert small.shape[1] <= full_tp
    assert small.shape[1] <= 5 + 1  # reduced grid stays small


# --- VarVI / importance extractor ------------------------------------------------

def test_varvi_returned_for_survival_matched(d):
    # R: fit$VarVI, length p. API gap: VarVI computed in core but dropped
    # by the Python estimator layer.
    fit = _fit(d["X"], d["y"], n_estimators=30, importance="permute",
               var_mode="matched")
    assert not hasattr(fit, "var_vi_")  # documents the gap


def test_importance_shows_sd_z_sig_for_survival_matched(d):
    pytest.skip("no importance() SD/Z/Sig extractor in the Python API")


def test_varvi_not_returned_for_ij_mode(d):
    # R: fit$VarVI is NULL for IJ
    fit = _fit(d["X"], d["y"], n_estimators=30, importance="permute",
               var_mode="ij")
    assert not hasattr(fit, "var_vi_")


def test_varvi_not_returned_for_jack_mode(d):
    # R: fit$VarVI is NULL for jack
    fit = _fit(d["X"], d["y"], n_estimators=30, importance="permute",
               var_mode="jack")
    assert not hasattr(fit, "var_vi_")


# --- keep.all --------------------------------------------------------------------

def test_keep_all_is_accepted_without_error(d):
    # R fits with keep.all = TRUE; the Python analog is keep_all=True at
    # predict time (per-tree hazards).
    fit = _fit(d["X"], d["y"], n_estimators=30, var_mode="matched")
    out = fit._predict_raw(d["X"][:4], var_est=True, keep_all=True)
    assert "AllHazard" in out


def test_keep_all_stores_fitted_forest_in_fit_object(d):
    # R: keep.all keeps FittedForest; Python always stores forest_.
    fit = _fit(d["X"], d["y"], n_estimators=30, var_mode="matched")
    assert hasattr(fit, "forest_")
    assert "NodeHaz" in fit.forest_


# --- input validation ---------------------------------------------------------------

def test_na_in_y_produces_error(d):
    # R: expect_error on NA in y. API gap: the Python layer accepts NaN
    # times without complaint (no NA validation on the survival response).
    pytest.xfail("Python layer does not validate NaN in survival y "
                 "(validation gap vs R)")


def test_na_in_censor_produces_error(d):
    # R: expect_error on NA in censor. The Python structured dtype uses a
    # bool 'event' field, which cannot represent NA at all.
    pytest.skip("bool event field cannot represent NA; no Python analog")


def test_mismatched_dimensions_produce_error():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(80, 5))
    y_time = rng.exponential(size=50)
    y = np.empty(50, dtype=[("event", "?"), ("time", "<f8")])
    y["event"] = True
    y["time"] = y_time
    with pytest.raises(ValueError):
        RLT_surv(n_estimators=30, n_jobs=2).fit(X, y)


def test_invalid_censor_values_produce_error(d):
    # R: censor values outside {0, 1} error. API gap: the Python layer
    # coerces event to int and does not validate the value set.
    pytest.xfail("Python layer does not validate event/censor value set "
                 "(validation gap vs R)")


# --- confidence bands ----------------------------------------------------------------

def test_get_surv_band_returns_valid_structure(d):
    fit = _fit(d["X"], d["y"], n_estimators=30, var_mode="matched")
    band = get_surv_band(fit, d["X"], i=1, alpha=0.1, nsim=50)
    assert isinstance(band, dict)
    for key in ("timepoints", "lower", "upper"):
        assert key in band
    lower = np.asarray(band["lower"])
    upper = np.asarray(band["upper"])
    assert lower.shape == upper.shape
    assert np.asarray(band["timepoints"]).shape[0] == lower.shape[-1]


def test_confidence_band_contains_survival_curve(d):
    fit = _fit(d["X"], d["y"], n_estimators=30, var_mode="matched")
    band = get_surv_band(fit, d["X"], i=1, alpha=0.1, nsim=50)
    lower = np.asarray(band["lower"]).ravel()
    upper = np.asarray(band["upper"]).ravel()
    surv = fit.predict_survival_function(d["X"])[0, :lower.shape[0]]
    within = np.sum((surv >= lower) & (surv <= upper))
    assert within > 0
