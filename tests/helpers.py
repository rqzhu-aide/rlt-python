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


# ---------------------------------------------------------------------------
# Ports of RLT's testthat/helper-synthetic-data.R
# ---------------------------------------------------------------------------

def generate_mixed_feature_regression(n=400, p=15, seed=1):
    """Port of generate_mixed_feature_regression().

    Half continuous, half categorical (3 levels: 0, 1, 2) features.  The true
    model depends on continuous variables X2..X6 and the first categorical
    variable (levels 1 and 2 have an effect).

    Returns a dict mirroring the R list: X (float ndarray, categorical columns
    kept as integer codes since the Python API has no per-column factor
    support), y, n, p, n_continuous, n_categorical, feature_names,
    true_model.
    """
    rng = np.random.default_rng(seed)

    # Ensure p is even for equal split
    if p % 2 != 0:
        p = p + 1

    n_continuous = p // 2
    n_categorical = p // 2

    X1 = rng.normal(size=(n, n_continuous))
    X2 = np.floor(rng.random((n, n_categorical)) * 3).astype(int)

    X = np.column_stack([X1, X2]).astype(float)
    feature_names = ([f"X{i + 1}" for i in range(n_continuous)]
                     + [f"Xcat{i + 1}" for i in range(n_categorical)])

    # True model: continuous variables X2-X6 + categorical Xcat1 in {1, 2}
    cont_sum = X[:, 1:min(6, n_continuous)].sum(axis=1)
    y = (1 + cont_sum + 2 * np.isin(X[:, n_continuous], [1, 2]).astype(float)
         + rng.normal(size=n))

    return {
        "X": X,
        "y": y,
        "n": n,
        "p": p,
        "n_continuous": n_continuous,
        "n_categorical": n_categorical,
        "feature_names": feature_names,
        "true_model": "y = 1 + X2 + X3 + X4 + X5 + X6 + 2*I(Xcat1 in {1,2}) + noise",
    }


def generate_simple_regression(n=200, p=10, seed=1):
    """Port of generate_simple_regression(): all-continuous features."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = 1 + X[:, 0] + 2 * X[:, 1] + rng.normal(size=n)
    return {
        "X": X,
        "y": y,
        "n": n,
        "p": p,
        "true_model": "y = 1 + X1 + 2*X2 + noise",
    }


def generate_classification_data(n=400, p=15, seed=1):
    """Port of generate_classification_data(): binary outcome."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    prob = 1.0 / (1.0 + np.exp(-(X[:, 0] + X[:, 1] - X[:, 2])))
    y = (rng.random(n) < prob).astype(int)
    return {
        "X": X,
        "y": y,
        "n": n,
        "p": p,
        "true_model": "logit(P(y=1)) = X1 + X2 - X3",
    }


def generate_survival_data(n=400, p=10, censoring_rate=0.3, seed=None):
    """Port of generate_survival_data(): right-censored survival data.

    Returns a dict with X, y (structured array with 'event'/'time'),
    censor (0/1), n, p, censoring_rate, true_model.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    X = rng.normal(size=(n, p))

    true_effect = X[:, 0] + 0.5 * X[:, 1]
    hazard = np.exp(true_effect)
    surv_time = rng.exponential(1.0 / hazard)

    if 0 < censoring_rate < 1:
        censor_time = rng.exponential(
            1.0 / (np.quantile(hazard, 0.5) * (1 - censoring_rate) / censoring_rate),
            size=n,
        )
    elif censoring_rate == 0:
        censor_time = np.full(n, np.inf)
    else:
        censor_time = np.zeros(n)

    time = np.minimum(surv_time, censor_time)
    censor = (surv_time <= censor_time).astype(int)

    y = np.empty(n, dtype=[("event", "?"), ("time", "<f8")])
    y["event"] = censor.astype(bool)
    y["time"] = time

    return {
        "X": X,
        "y": y,
        "censor": censor,
        "n": n,
        "p": p,
        "censoring_rate": censor.mean(),
        "true_model": "hazard = exp(X1 + 0.5*X2)",
    }
