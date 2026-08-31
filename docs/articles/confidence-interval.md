# Survival Confidence Band

This page demonstrates how to construct simultaneous confidence bands for
individual survival curves predicted by `rlt` — the Python port of the RLT
R package's survival confidence band article. The example uses the smoothed
covariance approach in `get_surv_band()`.

Two details matter for a sensible visualization:

- Target points should be inside the observed predictor domain, especially for the important variables.
- The displayed time range should avoid the far tail where very few training observations remain at risk.

## Data

We simulate right-censored survival data from a proportional hazards
model. The first and third predictors carry signal; the rest are noise.

```python
import numpy as np
from rlt import RLT_surv

rng = np.random.default_rng(2)

n = 200
p = 10
X = rng.normal(size=(n, p))

def xlink(x):
    return np.exp(x[:, 0] + x[:, 2] / 2)

FT = rng.exponential(1.0 / xlink(X))
CT = np.minimum(6, rng.exponential(1.0 / 0.25, size=n))

time = np.minimum(FT, CT)
event = (FT <= CT).astype(bool)

y = np.empty(n, dtype=[("event", "?"), ("time", "<f8")])
y["event"] = event
y["time"] = time
```

## Target Points

We choose two target points at interior quantiles of the important
variables `X1` and `X3`. The remaining variables are fixed at their
training medians.

```python
target_base = np.median(X, axis=0)
testX = np.tile(target_base, (2, 1))

testX[:, 0] = np.quantile(X[:, 0], [0.35, 0.65])
testX[:, 2] = np.quantile(X[:, 2], [0.35, 0.65])

np.round(testX[:, [0, 2]], 3)
```

```
array([[-0.384, -0.385],
       [ 0.315,  0.377]])
```

## Fit and Predict with Variance

We use `var_mode="matched"` for variance estimation. This prepares
covariance estimates on the cumulative hazard scale, which is the scale
used by the confidence-band calculation.

```python
fit = RLT_surv(
    n_estimators=1000, mtry=min(p, 10), min_samples_leaf=5,
    nsplit=3,
    resample_prob=0.8, resample_replace=False,
    importance="none", verbose=0,
    n_jobs=1,
    var_mode="matched",
    split_rule="logrank",
    random_state=2,
)
fit.fit(X, y)

S, cov = fit.predict_var(testX)
```

## Restrict to Supported Follow-up Times

The fitted model can return predictions over the observed event-time grid,
but the far right tail has little data support. For this example, we only
display the region `t <= 2`.

```python
risk_set = np.array([np.sum(time >= t) for t in fit.timepoints_])
plot_time_max = 2
plot_id = fit.timepoints_ <= plot_time_max

if not plot_id.any():
    plot_id = np.zeros_like(plot_id, dtype=bool)
    plot_id[0] = True

print(f"Plot time range: t <= {plot_time_max}")
print(f"Minimum at-risk in plotted range: {int(risk_set[plot_id].min())}")
print(f"Number of plotted timepoints: {int(plot_id.sum())}")
```

```
Plot time range: t <= 2
Minimum at-risk in plotted range: 33
Number of plotted timepoints: 136
```

## Simultaneous Confidence Band

`get_surv_band()` with `approach="smoothed"` constructs a simultaneous
confidence band using a smoothed low-rank approximation of the covariance
matrix.

```python
from rlt import get_surv_band

SurvBand = get_surv_band(fit, testX, alpha=0.05, approach="smoothed",
                         k_rank=10, nsim=5000, seed=0)
```

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
titles = ["Interior lower-risk target", "Interior higher-risk target"]

for i, ax in enumerate(axes):
    tp = fit.timepoints_[plot_id]
    S_i = np.clip(S[i, plot_id], 0, 1)
    truth = np.exp(-xlink(testX)[i] * tp)

    b_lower = np.clip(SurvBand["lower"][i][plot_id], 0, 1)
    b_upper = np.clip(SurvBand["upper"][i][plot_id], 0, 1)

    ax.fill_between(tp, b_lower, b_upper, color="#3b82f6", alpha=0.25,
                    label="Simultaneous 95% Band")
    ax.plot(tp, truth, color="#E41A1C", lw=2, ls="--", label="True S(t)")
    ax.step(tp, S_i, where="post", color="black", lw=2, label="Estimated S(t)")

    ax.set_title(titles[i])
    ax.set_xlabel("Time")
    ax.set_ylim(0, 1)

axes[0].set_ylabel("Survival Probability")
axes[1].legend(loc="lower left", fontsize=8)
plt.tight_layout()
plt.show()
```

## Optional Rank Selection

The smoothed approach can also choose the rank by cumulative eigenvalue
proportion instead of a fixed `k_rank`:

```python
SurvBand_prop = get_surv_band(
    fit, testX, alpha=0.05,
    approach="smoothed",
    k_mode="proportion",
    k_prop=0.95,
)
```

You can increase `nsim` for stability at the cost of runtime.
