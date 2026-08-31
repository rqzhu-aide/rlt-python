"""Port of RLT testthat/test-survival-lc.R (survival linear combination).

Param map: ntrees->n_estimators, linear.comb->linear_comb,
linear.comb.method->linear_comb_method, split.rule->split_rule,
resample.replace->resample_replace, ncores->n_jobs, seed->random_state.
Survival response: structured array with fields ("event", "time").
"""
import numpy as np
import pytest

from rlt import RLT_surv
from helpers import generate_survival_data


def _fit(X, y, **kw):
    kw.setdefault("n_jobs", 2)
    return RLT_surv(**kw).fit(X, y)


@pytest.fixture()
def d():
    return generate_survival_data(n=80, p=10, seed=1)


@pytest.mark.parametrize("rule", ["logrank", "suplogrank", "coxgrad"])
def test_lc_split_rules_work(d, rule):
    _fit(d["X"], d["y"], n_estimators=30, linear_comb=3, split_rule=rule)


def test_lc_default_split_rule_is_logrank(d):
    _fit(d["X"], d["y"], n_estimators=30, linear_comb=3)
    # default resolves to logrank (code 1)
    assert _fit(d["X"], d["y"], n_estimators=30, linear_comb=3).params_.split_rule == 1


def test_lc_naive_method_works(d):
    _fit(d["X"], d["y"], n_estimators=30, linear_comb=3,
         linear_comb_method="naive")


def test_lc_prediction_works(d):
    fit = _fit(d["X"], d["y"], n_estimators=30, linear_comb=3)
    sf = fit.predict_survival_function(d["X"])
    S = np.asarray(sf)
    assert S.ndim == 2
    assert S.shape[0] == d["n"]


def test_lc_stores_splitload_in_forest(d):
    fit = _fit(d["X"], d["y"], n_estimators=30, linear_comb=3)
    assert "SplitLoad" in fit.forest_
    # ragged per-tree list of loading vectors
    loads = fit.forest_["SplitLoad"]
    assert len(loads) == 30
    assert all(np.asarray(t).size > 0 for t in loads)


def test_lc_reproducibility_with_same_seed(d):
    fit1 = _fit(d["X"], d["y"], n_estimators=30, linear_comb=3,
                random_state=42, n_jobs=1)
    fit2 = _fit(d["X"], d["y"], n_estimators=30, linear_comb=3,
                random_state=42, n_jobs=1)
    np.testing.assert_array_equal(fit1.oob_hazard_, fit2.oob_hazard_)
    assert fit1.oob_error_ == fit2.oob_error_


def test_lc_importance_works(d):
    # R: importance = TRUE -> fit$VarImp has length p
    fit = _fit(d["X"], d["y"], n_estimators=30, linear_comb=3,
               importance="permute")
    varimp = np.asarray(fit.varimp_)
    assert varimp.shape == (d["p"],)
    assert varimp.size == d["p"]
    assert np.isfinite(varimp).all()


def test_lc_with_subsampling_works(d):
    _fit(d["X"], d["y"], n_estimators=30, linear_comb=3,
         resample_replace=False)


def test_lc_nsplit_zero_works(d):
    _fit(d["X"], d["y"], n_estimators=30, linear_comb=3, nsplit=0)


def test_lc_nsplit_positive_works(d):
    _fit(d["X"], d["y"], n_estimators=30, linear_comb=3, nsplit=3)


def test_lc_with_alpha_works(d):
    _fit(d["X"], d["y"], n_estimators=30, linear_comb=3, alpha=0.2)
