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
array([ 1.010e-01,  7.940e-01,  1.690e-01,  4.270e-01, -2.000e-03,
        4.014e+00, -1.500e-02,  1.000e-03,  7.000e-03,  6.800e-02])
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
array([-5.00e-02,  6.96e-01,  8.00e-02,  5.45e-01, -1.90e-02,  4.19e+00,
        3.00e-03, -6.00e-02,  0.00e+00,  2.10e-02])
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

## Variance of variable importance (matched forests)

VI estimates are themselves noisy. Fitting with `var_mode="matched"`
(mixed with either importance option) makes the forest also return the
per-variable variance of the VI estimates, so you can judge which
importances are actually distinguishable from noise — the Python port of
R's `importance(fit)` output.

```python
fit_vi = RLT_reg(
    n_estimators=ntrees,
    mtry=mtry,
    min_samples_leaf=nmin,
    importance="distribute",
    var_mode="matched",   # per-variable variance of VI
    n_jobs=1,
    random_state=1,
)
fit_vi.fit(trainX, trainY)

# (p,) variance of each VI estimate
np.round(fit_vi.var_vi_, 4)
```

```
array([ 9.500e-03, -8.670e-02,  1.000e-04,  7.000e-02,  1.830e-02,
        1.366e-01, -7.000e-04, -2.300e-03,  9.300e-03,  3.140e-02])
```

The friendlier view is `importance_table()`, which prints `Variable`,
`VI`, and — when matched variance is available — the standard deviation
`SD = sqrt(VarVI)`, the standardized statistic `Z = VI / SD`, and a
significance code (`***` for |Z| ≥ 2.58, `**` for ≥ 1.96, `*` for
≥ 1.64). The layout mirrors R's `print.importance.RLT`:

```python
fit_vi.importance_table()
```

```
Variable             VI           SD          Z  Sig
----------------------------------------------------------
V1              -0.0130     0.097625      -0.13
V2               0.6622           NA         NA
V3               0.1547     0.007742      19.98  ***
V4               0.4933     0.264541       1.86  *
V5              -0.0993     0.135343      -0.73
V6               4.1172     0.369655      11.14  ***
V7              -0.0187           NA         NA
V8              -0.0093           NA         NA
V9              -0.0358     0.096533      -0.37
V10              0.0322     0.177089       0.18

Note: 3 variable(s) with negative variance estimate (SD, Z shown as NA)
```

Two things to note in this output:

- **Negative variance estimates.** The matched U-statistic variance can
  come out negative for noise variables. Exactly like R, such variables
  get `NA` for `SD` and `Z` and an empty significance code, and the
  table appends a trailing `Note` counting them. A negative variance
  estimate is itself informative: it flags variables with no stable
  importance signal.
- **Without matched variance**, the table simply has no `SD` / `Z` /
  `Sig` columns — `importance_table()` works for any
  `importance != "none"` fit and prints just `Variable` and `VI`.

The same table is available through the module-level function
`rlt.importance(model)`, mirroring R's `importance(fit)` idiom (the
estimator attribute `.importance` holds the constructor parameter
string, so the method could not take that name):

```python
import rlt

rlt.importance(fit_vi)   # same table as fit_vi.importance_table()
```

!!! note
    `var_mode="matched"` implies the matched-forest preset: subsampling
    without replacement at 50% and an even tree count. When importance is
    requested, use `importance="distribute"` — permutation importance
    (`"permute"`) is left as-is, but distributed importance is what the
    variance machinery in the core expects.
