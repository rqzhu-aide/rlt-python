"""Port of RLT tests/testthat/test-rlt-basic.R.

Tests for Reinforcement Learning Trees (embedded model) feature.
"""

import numpy as np
import pytest

from rlt import RLT_reg


def _egg_tray_data(n, p_noise, seed):
    """x1*x2 interaction ("egg tray") model plus noise columns."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    X_noise = rng.normal(size=(n, p_noise))
    X = np.column_stack([x1, x2, X_noise])
    y = x1 * x2 + 0.5 * rng.normal(size=n)
    return X, y


def test_rlt_identifies_x1_x2_as_most_important():
    # set.seed(42); n <- 80; p_noise <- 8
    X, y = _egg_tray_data(n=80, p_noise=8, seed=42)

    # RLT(X, y, ntrees = 30, ncores = 2, verbose = FALSE,
    #     embed.ntrees = 20, importance = TRUE)
    fit = RLT_reg(
        n_estimators=30,
        n_jobs=2,
        verbose=0,
        embed_ntrees=20,
        importance="permute",
        reinforcement=True,
        random_state=42,
    ).fit(X, y)

    # top2 <- colnames(X)[order(fit$VarImp, decreasing = TRUE)[1:2]]
    # x1 -> column 0, x2 -> column 1
    var_imp = fit.feature_importances_
    top2 = np.argsort(var_imp)[::-1][:2]
    assert 0 in top2
    assert 1 in top2


def test_rlt_muting_reduces_noise_variable_consideration():
    # set.seed(42); n <- 300; p_noise <- 8
    X, y = _egg_tray_data(n=300, p_noise=8, seed=42)

    # RLT(X, y, ntrees = 100, ncores = 2, verbose = FALSE,
    #     embed.ntrees = 50, embed.mtry = 2, importance = TRUE)
    fit = RLT_reg(
        n_estimators=100,
        n_jobs=2,
        verbose=0,
        embed_ntrees=50,
        embed_mtry=2,
        importance="permute",
        reinforcement=True,
        random_state=42,
    ).fit(X, y)

    var_imp = fit.feature_importances_
    # expect_true(all(!is.na(fit$VarImp)))
    assert np.all(np.isfinite(var_imp))

    # Signal VI (x1, x2) should be higher than noise mean
    signal_vi = var_imp[0:2]
    noise_vi = var_imp[2:10]
    assert np.mean(signal_vi) > np.mean(noise_vi)
