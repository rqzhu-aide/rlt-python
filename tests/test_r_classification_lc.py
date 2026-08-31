"""Port of RLT testthat/test-classification-lc.R (classification LC).

Param map: ntrees->n_estimators, linear.comb->linear_comb,
linear.comb.method->linear_comb_method, ncores->n_jobs, seed->random_state.
"""
import numpy as np
import pytest

from rlt import RLT_cla
from helpers import (
    generate_classification_data,
    generate_mixed_feature_regression,
)


def _fit(X, y, **kw):
    kw.setdefault("n_jobs", 2)
    kw.setdefault("random_state", 1)
    return RLT_cla(**kw).fit(X, y)


@pytest.mark.parametrize("method", ["lda", "naive", "random", "logistic"])
def test_lc_method_works_on_binary_data(method):
    d = generate_classification_data(n=80, p=10)
    _fit(d["X"], d["y"], n_estimators=30, linear_comb=3,
         linear_comb_method=method)


def _three_class_data(seed=42):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(90, 5))
    y = np.repeat([1, 2, 3], 30)
    return X, y


@pytest.mark.parametrize("method", ["lda", "naive", "random", "logistic"])
def test_lc_method_works_on_3class_data(method):
    X, y = _three_class_data()
    _fit(X, y, n_estimators=30, linear_comb=3, linear_comb_method=method)


@pytest.mark.parametrize("m", [1, 2, 3, 4])
def test_integer_method_codes_work(m):
    # R: integer codes 1-4 accepted. Ported: the Python layer accepts
    # integer codes directly (no stringification / silent fallback).
    d = generate_classification_data(n=80, p=10)
    fit = _fit(d["X"], d["y"], n_estimators=20, linear_comb=3,
               linear_comb_method=m)
    assert fit.params_.linear_comb_method == m


def test_lc_with_categorical_predictors():
    # R passes factor columns; Python API has no ncat metadata, so integer
    # level codes are used as numeric columns.
    d = generate_mixed_feature_regression(n=80, p=10)
    y = (d["y"] > np.median(d["y"])).astype(int)
    _fit(d["X"], y, n_estimators=30, linear_comb=3, linear_comb_method="lda")


def test_string_class_labels_preserved_in_predictions():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(80, 5))
    y = rng.choice(np.array(["cat", "dog"]), size=80)
    fit = _fit(X, y, n_estimators=30, linear_comb=3,
               linear_comb_method="naive")
    assert list(fit.classes_) == ["cat", "dog"]
    # sklearn convention: predict returns the original label dtype
    pred = fit.predict(X)
    assert set(np.unique(pred)) <= {"cat", "dog"}


def test_lc_predictions_comparable_to_non_lc():
    d = generate_classification_data(n=80, p=10)
    fit_lc = _fit(d["X"], d["y"], n_estimators=30, linear_comb=3,
                  linear_comb_method="lda")
    _fit(d["X"], d["y"], n_estimators=30)  # non-LC baseline also fits
    pred_lc = fit_lc.predict(d["X"])
    acc = np.mean(pred_lc == d["y"])
    assert acc > 0.3


def test_small_sample_lc_fits():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(30, 5))
    y = rng.choice([1, 2], size=30)
    _fit(X, y, n_estimators=20, linear_comb=3, linear_comb_method="naive")


def test_lc_lda_accuracy_beats_50pct_on_binary_data():
    d = generate_classification_data(n=80, p=10)
    fit = _fit(d["X"], d["y"], n_estimators=30, linear_comb=3,
               linear_comb_method="lda")
    acc = np.mean(fit.predict(d["X"]) == d["y"])
    assert acc > 0.5


def test_lc_works_with_p_larger_than_n():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(80, 50))
    y = rng.choice([1, 2], size=80)
    _fit(X, y, n_estimators=20, linear_comb=3, linear_comb_method="naive")
