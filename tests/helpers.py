"""Shared synthetic data generators, mirroring RLT's testthat helpers."""

import numpy as np


def make_regression(n=300, p=5, noise=1.0, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = X[:, 0] + X[:, 1] * X[:, 1] + noise * rng.normal(size=n)
    return X, y


def make_classification(n=300, p=5, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    eta = X[:, 0] - X[:, 1]
    y = (eta + rng.normal(size=n) * 0.5 > 0).astype(int)
    return X, y


def make_survival(n=300, p=5, censoring=0.3, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    lam = np.exp(X[:, 0] + 0.5 * X[:, 1])
    time = rng.exponential(1.0 / lam)
    ctime = rng.exponential(np.quantile(1.0 / lam, 1 - censoring))
    event = (time <= ctime).astype(int)
    time = np.minimum(time, ctime)
    y = np.empty(n, dtype=[("event", "?"), ("time", "<f8")])
    y["event"] = event.astype(bool)
    y["time"] = time
    return X, y
