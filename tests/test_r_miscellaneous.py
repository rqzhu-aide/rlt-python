"""Port of RLT tests/testthat/test-miscellaneous.R."""

import numpy as np
import pytest

from rlt import RLT_reg, RLT_cla
from tests.helpers import generate_simple_regression, generate_classification_data


# --- Reproducibility ---

def test_same_seed_produces_identical_results_across_models():
    d = generate_simple_regression(n=80, p=10)
    fit1 = RLT_reg(n_estimators=30, random_state=42, n_jobs=1, verbose=0).fit(
        d["X"], d["y"]
    )
    fit2 = RLT_reg(n_estimators=30, random_state=42, n_jobs=1, verbose=0).fit(
        d["X"], d["y"]
    )
    np.testing.assert_array_equal(fit1.oob_prediction_, fit2.oob_prediction_)


def test_different_seeds_produce_different_results():
    d = generate_simple_regression(n=80, p=10)
    fit1 = RLT_reg(n_estimators=30, random_state=1, n_jobs=1, verbose=0).fit(
        d["X"], d["y"]
    )
    fit2 = RLT_reg(n_estimators=30, random_state=2, n_jobs=1, verbose=0).fit(
        d["X"], d["y"]
    )
    assert not np.array_equal(fit1.oob_prediction_, fit2.oob_prediction_)


def test_reproducibility_works_with_linear_combinations():
    d = generate_simple_regression(n=80, p=10)
    fit1 = RLT_reg(
        n_estimators=30,
        random_state=42,
        n_jobs=1,
        verbose=0,
        linear_comb=3,
        linear_comb_method="naive",
    ).fit(d["X"], d["y"])
    fit2 = RLT_reg(
        n_estimators=30,
        random_state=42,
        n_jobs=1,
        verbose=0,
        linear_comb=3,
        linear_comb_method="naive",
    ).fit(d["X"], d["y"])
    np.testing.assert_array_equal(fit1.oob_prediction_, fit2.oob_prediction_)


# --- Core counts ---

def test_single_core_njobs_1_works():
    d = generate_simple_regression(n=80, p=10)
    RLT_reg(n_estimators=30, n_jobs=1, verbose=0).fit(d["X"], d["y"])


def test_all_cores_njobs_all_works():
    # R: ncores = 0 means all cores; Python: n_jobs = -1
    d = generate_simple_regression(n=80, p=10)
    RLT_reg(n_estimators=30, n_jobs=-1, verbose=0).fit(d["X"], d["y"])


def test_specific_core_count_njobs_2_works():
    d = generate_simple_regression(n=80, p=10)
    RLT_reg(n_estimators=30, n_jobs=2, verbose=0).fit(d["X"], d["y"])


def test_multicore_run_completes_and_returns_valid_predictions():
    d = generate_simple_regression(n=80, p=10)
    fit = RLT_reg(n_estimators=30, random_state=42, n_jobs=2, verbose=0).fit(
        d["X"], d["y"]
    )
    assert np.asarray(fit.oob_prediction_).shape == (d["n"],)
    assert np.isfinite(fit.oob_error_)


# --- Kernel ---

def test_self_kernel_matrix_computation_works():
    d = generate_simple_regression(n=80, p=10)
    fit = RLT_reg(n_estimators=30, n_jobs=2, verbose=0).fit(d["X"], d["y"])
    K = fit.forest_kernel(d["X"])
    # R: K is returned as an RLT object with $Kernel; Python returns ndarray
    assert isinstance(K, np.ndarray)
    assert K.shape == (d["n"], d["n"])


def test_cross_kernel_matrix_computation_works():
    d = generate_simple_regression(n=80, p=10)
    fit = RLT_reg(n_estimators=30, n_jobs=2, verbose=0).fit(
        d["X"][:60], d["y"][:60]
    )
    K = fit.forest_kernel(d["X"][:60], d["X"][60:])
    assert K.shape == (60, 20)


def test_kernel_properties_are_correct():
    d = generate_simple_regression(n=80, p=10)
    fit = RLT_reg(n_estimators=30, n_jobs=2, verbose=0).fit(d["X"], d["y"])
    K = fit.forest_kernel(d["X"])
    # Self-kernel should be symmetric
    np.testing.assert_allclose(K, K.T)
    # Diagonal should equal ntrees (each obs in exactly 1 node per tree)
    np.testing.assert_allclose(np.diag(K), np.full(d["n"], 30))


def test_kernel_based_prediction():
    d = generate_simple_regression(n=80, p=10)
    train_idx = np.arange(60)
    test_idx = np.arange(60, 80)
    fit = RLT_reg(n_estimators=30, n_jobs=2, verbose=0).fit(
        d["X"][train_idx], d["y"][train_idx]
    )
    K = fit.forest_kernel(d["X"][train_idx], d["X"][test_idx])
    kernel_pred = K.T @ d["y"][train_idx] / K.sum(axis=0)
    direct_pred = fit.predict(d["X"][test_idx])
    # Kernel-weighted prediction should be correlated with direct prediction
    assert np.corrcoef(kernel_pred, direct_pred)[0, 1] > 0.5


def test_classification_forest_kernel_works():
    d = generate_classification_data(n=80, p=10)
    fit = RLT_cla(n_estimators=30, n_jobs=2, verbose=0).fit(d["X"], d["y"])
    K = fit.forest_kernel(d["X"])
    assert K.shape == (d["n"], d["n"])


def test_classification_cross_kernel_works():
    d = generate_classification_data(n=80, p=10)
    fit = RLT_cla(n_estimators=30, n_jobs=2, verbose=0).fit(
        d["X"][:60], d["y"][:60]
    )
    K = fit.forest_kernel(d["X"][:60], d["X"][60:])
    assert K.shape == (60, 20)


# --- Small data ---

def test_reproducibility_with_very_small_data():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(10, 2))
    y = X[:, 0] + rng.normal(size=10)
    fit1 = RLT_reg(n_estimators=10, random_state=1, n_jobs=1, verbose=0).fit(X, y)
    fit2 = RLT_reg(n_estimators=10, random_state=1, n_jobs=1, verbose=0).fit(X, y)
    np.testing.assert_array_equal(fit1.oob_prediction_, fit2.oob_prediction_)


def test_kernel_with_single_tree():
    d = generate_simple_regression(n=80, p=10)
    fit = RLT_reg(n_estimators=1, n_jobs=2, verbose=0).fit(d["X"], d["y"])
    K = fit.forest_kernel(d["X"])
    assert K.shape == (d["n"], d["n"])
