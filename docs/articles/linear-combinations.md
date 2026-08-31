# RLT with Linear Combinations

Linear-combination splits allow each internal node to split on a weighted
combination of several variables rather than a single predictor. This can
help when the signal is carried by a direction such as `X1 + X2` instead of
one coordinate at a time.

The example below uses regression. The same idea can also be applied to
classification and survival forests by using `RLT_cla` or `RLT_surv` and
choosing a model-appropriate `linear_comb_method`.

## Regression Example

```python
import numpy as np

rng = np.random.default_rng(3)

n = 160
p = 6
X = rng.normal(size=(n, p))

y = 1 + X[:, 0] + X[:, 1] + 0.5 * X[:, 2] + rng.normal(size=n, scale=0.5)

train_id = np.arange(120)
trainX, trainY = X[:120], y[:120]
testX, testY = X[120:], y[120:]
```

Fit a regression forest that considers three-variable linear combinations
at a split.

```python
from rlt import RLT_reg

fit_lc = RLT_reg(
    n_estimators=200,
    min_samples_leaf=5,
    mtry=3,
    nsplit=3,
    linear_comb=3,
    linear_comb_method="sir",
    n_jobs=1,
    random_state=3,
)
fit_lc.fit(trainX, trainY)

print(fit_lc)
```

```
RLT_reg(linear_comb=3, linear_comb_method='sir', mtry=3, n_jobs=1,
        n_estimators=200, nsplit=3, random_state=3)
```

## Prediction

```python
test_pred = fit_lc.predict(testX)

print(f"Test MSE: {np.mean((test_pred - testY) ** 2):.4f}")
print(f"Linear combination size: {fit_lc.linear_comb}")
```

```
Test MSE: 0.3014
Linear combination size: 3
```

## Other Model Types

For classification, use `RLT_cla` with methods such as `"lda"`, `"naive"`,
`"random"`, or `"logistic"`.

For survival, use `RLT_surv` with methods such as `"coxph"` or `"naive"`.
