# RLT_cla

RLT classification forest — sklearn-compatible estimator porting the RLT R
package's `RLT(model = "classification")`.

`RLT_cla` grows a random forest of classification trees. Labels may be
integers, strings, or any one-dimensional array of distinct values;
`classes_` records the sorted unique labels and `predict` returns the
original label type.

```python
import numpy as np
from rlt import RLT_cla

rng = np.random.default_rng(42)
X = rng.normal(size=(100, 5))
y = rng.binomial(1, 1 / (1 + np.exp(-2 * X[:, 0])))

model = RLT_cla(n_estimators=500, random_state=42)
model.fit(X, y)                # string labels fine
prob = model.predict_proba(X)  # (n, n_classes)
```

## Constructor

```python
RLT_cla(
    n_estimators=500,
    mtry=None,
    min_samples_leaf=5,
    nsplit=0,
    resample_replace=True,
    resample_prob=None,
    importance="none",
    resample_track=False,
    var_mode="none",
    linear_comb=1,
    linear_comb_method="default",
    alpha=0,
    reinforcement=False,
    embed_ntrees=50,
    embed_mtry=0.5,
    embed_nmin=5,
    embed_nsplit=3,
    embed_resample_replace=True,
    embed_resample_prob=0.9,
    embed_mute=0,
    embed_protect=None,
    embed_threshold=0.25,
    n_jobs=-1,
    verbose=0,
    random_state=None,
)
```

### Parameters

All parameters match [`RLT_reg`](rlt_reg.md) — see that page for the full
table — with one classification-specific difference:

| Parameter | Type / values | Default | Description |
|---|---|---|---|
| `linear_comb_method` | "default", "lda", "naive", "random", "logistic" | "default" | How combination loadings are computed ("lda" is the classification default). |

## Methods

| Method | Returns | Description |
|---|---|---|
| `fit(X, y, sample_weight=None, var_prob=None)` | `self` | Grow the forest. |
| `predict(X)` | labels | Predicted class labels (original label type). |
| `predict_proba(X)` | `(n, n_classes)` ndarray | Class-membership probabilities. |
| `predict_log_proba(X)` | `(n, n_classes)` ndarray | Log-probabilities. |
| `forest_kernel(X1, X2=None, vs_train=False)` | integer ndarray | Forest similarity kernel. |
| `get_one_tree(tree_id)` | dict of ndarrays | Raw arrays of tree `tree_id` (0-indexed). |
| `score(X, y)` | float | Mean accuracy (sklearn convention). |

## Fitted attributes

| Attribute | Shape | Description |
|---|---|---|
| `classes_` | `(n_classes,)` | Sorted unique labels seen at fit time. |
| `forest_` | dict | The full fitted forest. |
| `oob_class_` | `(n,)` | Out-of-bag predicted classes (integer codes). |
| `oob_prob_` | `(n, n_classes)` | Out-of-bag class probabilities. |
| `oob_error_` | float | Out-of-bag misclassification rate. |
| `feature_importances_` | `(p,)` | Variable importance (when `importance != "none"`). |
| `obstrack_` | `(n, n_estimators)` | Inbag-count matrix (when tracked). |
| `n_features_in_` | int | Number of predictors. |

## Examples

```python
import numpy as np
from rlt import RLT_cla

rng = np.random.default_rng(0)
X = rng.normal(size=(150, 5))
prob = 1 / (1 + np.exp(-(2 * X[:, 0] - 1)))
y = rng.binomial(1, prob)

model = RLT_cla(n_estimators=500, importance="permute", random_state=0)
model.fit(X, y)
model.predict_proba(X[:3])     # (3, 2)
model.predict(X[:3])           # e.g. array([1, 0, 1])

# String labels
y_str = np.array(["neg", "pos"])[y]
m2 = RLT_cla(n_estimators=100, random_state=0).fit(X, y_str)
m2.classes_                    # array(['neg', 'pos'])
```

## See also

- [Classification tutorial](../tutorials/classification-tutorial.md)
- [`RLT_reg`](rlt_reg.md) for the full parameter reference
