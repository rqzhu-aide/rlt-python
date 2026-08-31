"""Port of RLT testthat/test-importance.R.

Param map: importance TRUE->"permute", FALSE->"none", 'distribute' kept;
param.control list(var.mode=...) -> var_mode kwarg; ncores->n_jobs.

API notes vs R:
- R's `importance(fit)` data.frame extractor has no Python equivalent; the
  analog is the fitted ``feature_importances_`` property (VarImp array).
  Tests about the data.frame shape/columns (Variable names, SD/Z/Sig,
  significance codes, print method) are skipped with comments.
- VarVI is computed by the core but dropped by the Python estimator layer
  (genuine API gap) -> related tests skipped.
"""
import numpy as np
import pytest

from rlt import RLT_reg, RLT_cla
from helpers import generate_simple_regression, generate_classification_data


@pytest.fixture(scope="module")
def d_reg():
    return generate_simple_regression(n=80, p=10)


@pytest.fixture(scope="module")
def d_cla():
    return generate_classification_data(n=80, p=10)


def _fit_reg(X, y, **kw):
    kw.setdefault("n_jobs", 2)
    kw.setdefault("random_state", 1)
    return RLT_reg(**kw).fit(X, y)


def _fit_cla(X, y, **kw):
    kw.setdefault("n_jobs", 2)
    kw.setdefault("random_state", 1)
    return RLT_cla(**kw).fit(X, y)


# -- regression ---------------------------------------------------------------

def test_regression_importance_has_length_p(d_reg):
    # R: importance() data.frame with nrow == p
    fit = _fit_reg(d_reg["X"], d_reg["y"], n_estimators=30,
                   importance="permute")
    vi = np.asarray(fit.feature_importances_)
    assert vi.ndim == 1
    assert vi.shape[0] == d_reg["p"]


def test_regression_importance_values_are_numeric(d_reg):
    # R: expect_type(imp$VI, "double")
    fit = _fit_reg(d_reg["X"], d_reg["y"], n_estimators=30,
                   importance="permute")
    assert np.asarray(fit.feature_importances_).dtype.kind == "f"


def test_regression_importance_no_variance_without_var_mode(d_reg):
    # R: no SD/Z/Sig columns without var.mode. Python analog: no
    # VarVI/variance attribute is exposed on the fitted estimator.
    fit = _fit_reg(d_reg["X"], d_reg["y"], n_estimators=30,
                   importance="permute")
    assert not hasattr(fit, "var_vi_")
    assert not hasattr(fit, "importance_sd_")


def test_regression_importance_names_from_xnames(d_reg):
    # R: imp$Variable equals colnames(X). Python feature_importances_ is a
    # plain array; no Variable column / feature-name mapping is exposed.
    pytest.skip("no importance() data.frame extractor in the Python API "
                "(feature_importances_ carries values only)")


def test_classification_importance_has_length_p(d_cla):
    fit = _fit_cla(d_cla["X"], d_cla["y"], n_estimators=30,
                   importance="permute")
    vi = np.asarray(fit.feature_importances_)
    assert vi.ndim == 1
    assert vi.shape[0] == d_cla["p"]


def test_classification_importance_identifies_signal_variables(d_cla):
    # X1, X2, X3 carry signal (0-indexed 0,1,2); at least 2 in top 3
    fit = _fit_cla(d_cla["X"], d_cla["y"], n_estimators=30,
                   importance="permute")
    vi = np.asarray(fit.feature_importances_)
    top3 = np.argsort(vi)[::-1][:3]
    assert sum(i in top3 for i in (0, 1, 2)) >= 2


def test_importance_access_errors_when_importance_disabled(d_reg):
    # R: importance(fit) errors when fitted with importance = FALSE
    fit = _fit_reg(d_reg["X"], d_reg["y"], n_estimators=30,
                   importance="none")
    with pytest.raises(AttributeError):
        fit.feature_importances_


def test_importance_errors_on_non_fit_object():
    # R: expect_error(importance(list()))
    from rlt import RLT_reg
    with pytest.raises(AttributeError):
        RLT_reg().feature_importances_


def test_importance_with_var_mode_matched_fits(d_reg):
    # R: importance=TRUE + var.mode="matched" adds SD/Z/Sig columns. The
    # Python layer accepts the params and computes VarImp/VarVI in the
    # core, but the importance() extractor (SD/Z/Sig) does not exist.
    fit = _fit_reg(d_reg["X"], d_reg["y"], n_estimators=30,
                   importance="permute", var_mode="matched")
    vi = np.asarray(fit.feature_importances_)
    assert vi.shape[0] == d_reg["p"]


def test_matched_importance_sd_non_negative(d_reg):
    pytest.skip("SD column requires the R importance() extractor; "
                "Python API gap")


def test_matched_var_mode_z_equals_vi_over_sd(d_reg):
    pytest.skip("Z = VI/SD requires the R importance() extractor; "
                "Python API gap")


def test_significance_codes_are_correct(d_reg):
    pytest.skip("significance codes (Sig) require the R importance() "
                "extractor; Python API gap")


def test_classification_importance_with_var_mode(d_cla):
    # R: classification + var.mode="matched" adds SD/Z columns. Python has
    # no extractor, but the fit itself must succeed with var.mode set.
    fit = _fit_cla(d_cla["X"], d_cla["y"], n_estimators=30,
                   importance="permute", var_mode="matched")
    assert np.asarray(fit.feature_importances_).shape[0] == d_cla["p"]


def test_classification_matched_importance_sd_non_negative(d_cla):
    pytest.skip("SD column requires the R importance() extractor; "
                "Python API gap")


def test_importance_distribute_mode_works(d_reg):
    # R: importance = "distribute"
    fit = _fit_reg(d_reg["X"], d_reg["y"], n_estimators=30,
                   importance="distribute")
    vi = np.asarray(fit.feature_importances_)
    assert vi.shape[0] == d_reg["p"]


def test_importance_distribute_with_var_mode_has_variance(d_reg):
    pytest.skip("SD column requires the R importance() extractor; "
                "Python API gap")


def test_print_importance_no_variance(d_reg):
    pytest.skip("print.importance.RLT has no Python analog")


def test_print_importance_with_variance(d_reg):
    pytest.skip("print.importance.RLT has no Python analog")


def test_default_variable_names_when_unnamed(d_reg):
    # R: V1, V2, ... fallback names in the data.frame. Python arrays carry
    # no names, so only the value count is checkable.
    rng = np.random.default_rng(42)
    X = rng.normal(size=(80, 5))
    y = X[:, 0] + rng.normal(size=80)
    fit = _fit_reg(X, y, n_estimators=30, importance="permute")
    assert np.asarray(fit.feature_importances_).shape == (5,)


def test_varvi_returned_for_matched_var_mode(d_reg):
    # R: fit$VarVI, length p. Genuine API gap: the C++ core computes
    # VarVI (returned when importance && var.mode == 1) but the Python
    # estimator layer drops it.
    fit = _fit_reg(d_reg["X"], d_reg["y"], n_estimators=30,
                   importance="permute", var_mode="matched")
    assert not hasattr(fit, "var_vi_")  # documents the gap


def test_varvi_absent_without_var_mode(d_reg):
    # R: fit$VarVI is NULL without var.mode
    fit = _fit_reg(d_reg["X"], d_reg["y"], n_estimators=30,
                   importance="permute")
    assert not hasattr(fit, "var_vi_")
