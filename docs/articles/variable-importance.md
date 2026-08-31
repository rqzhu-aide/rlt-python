# Variable Importance (Permutation and Distributed)

This page demonstrates how to obtain and visualize variable importance
(VI) from `rlt` — the Python port of the RLT R package's variable
importance article.

## Data

We generate continuous and categorical predictors with a continuous outcome.

```python
import numpy as np

# (Optional) For reproducibility in this tutorial only.
rng = np.random.default_rng(1)

# ---- Generate a small synthetic dataset ----
trainn = 80
testn = 20
n = trainn + testn
p = 10

# Continuous + categorical predictors (last half as categorical)
X1 = rng.normal(size=(n, p // 2))
X2 = (rng.uniform(size=(n, p // 2)) * 3).astype(int)  # integers 0, 1, 2
X = np.hstack([X1, X2])

# Continuous outcome with a simple signal + noise
y = (
    1
    + X[:, 1:6].sum(axis=1)
    + 2 * np.isin(X[:, p // 2], [1, 2])
    + rng.normal(size=n)
)

# Train / test split
trainX, trainY = X[:trainn], y[:trainn]
testX, testY = X[trainn:], y[trainn:]
```

## Option A - Permutation Importance

Set `importance="permute"` to enable permutation-based VI during training.

```python
from rlt import RLT_reg

# Minimal, sensible defaults
ntrees = 200
nmin = 5
mtry = p // 2
samplereplace = True
sampleprob = 0.80
rule = "best"
nsplit = 0 if rule == "best" else 3

fit_perm = RLT_reg(
    n_estimators=ntrees,
    mtry=mtry,
    min_samples_leaf=nmin,
    resample_prob=sampleprob,
    resample_replace=samplereplace,
    nsplit=nsplit,
    importance="permute",  # permutation-based VI
    n_jobs=1,
    random_state=1,
)
fit_perm.fit(trainX, trainY)

# VI vector lives in feature_importances_
np.round(fit_perm.feature_importances_, 3)
```

```
array([ 0.022,  0.824,  0.703,  0.964,  0.596,  3.375,  0.034,  0.016,
        0.007,  0.02 ])
```

```python
import matplotlib.pyplot as plt

# Simple visualization
fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(np.arange(p), fit_perm.feature_importances_)
ax.set_title("Permutation Variable Importance (rlt)")
ax.set_ylabel("Importance")
ax.set_xlabel("Predictors")
plt.show()
```

## Option B - Distributed Assignment Importance

This configuration assigns importance using distributed attribution with
OOB tracking. Use `importance="distribute"` for distributed assignment
importance. Unlike permutation importance, distributed importance works by
probabilistically routing OOB observations through the tree when a split
on the target variable is encountered. This requires sufficient OOB
samples per tree — avoid very high `resample_prob` with
`resample_replace=False`, which leaves too few OOB observations.

```python
fit_dist = RLT_reg(
    n_estimators=ntrees,
    mtry=mtry,
    min_samples_leaf=nmin,
    nsplit=nsplit,
    resample_prob=0.632,      # ~63.2% in-bag, ~36.8% OOB
    resample_replace=False,   # without replacement
    importance="distribute",  # distributed assignment VI
    n_jobs=1,
    random_state=1,
)
fit_dist.fit(trainX, trainY)

# VI vector lives in feature_importances_
np.round(fit_dist.feature_importances_, 3)
```

```
array([-0.038,  0.81 ,  0.443,  1.055,  0.261,  4.84 , -0.024, -0.07 ,
       -0.012,  0.009])
```

```python
fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(np.arange(p), fit_dist.feature_importances_)
ax.set_title("Distributed Variable Importance (rlt)")
ax.set_ylabel("Importance")
ax.set_xlabel("Predictors")
plt.show()
```

Both measures recover the design: variables 1–4 (indices 1:6) and
especially the categorical variable at index 5 carry the signal; the rest
are noise.
