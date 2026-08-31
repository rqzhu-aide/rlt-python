# Single Tree

This page shows how to inspect a single tree grown inside an `rlt` forest
using `get_one_tree()` — the Python port of the RLT R package's single-tree
article.

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

## Fit

```python
from rlt import RLT_reg

ntrees = 200
nmin = 5
mtry = p // 2
samplereplace = True
sampleprob = 0.80
rule = "best"
nsplit = 0 if rule == "best" else 3
importance = "permute"

fit = RLT_reg(
    n_estimators=ntrees,
    mtry=mtry,
    min_samples_leaf=nmin,
    resample_prob=sampleprob,
    resample_replace=samplereplace,
    nsplit=nsplit,
    importance=importance,
    n_jobs=1,
    random_state=1,
)
fit.fit(trainX, trainY)
```

## Inspect one tree

Use `get_one_tree(tree_id)` to inspect a single tree.
Choose `tree_id` from 0 to `n_estimators - 1` (0-indexed, unlike R's
1-based indexing). Here we take the first tree:

```python
tree = fit.get_one_tree(0)
tree.keys()
```

```
dict_keys(['SplitVar', 'SplitValue', 'LeftNode', 'RightNode', 'NodeWeight', 'NodeAve'])
```

The arrays are aligned by node index:

| Array | Meaning |
|-------|---------|
| `SplitVar` | Variable index split at this node; -1 for terminal nodes |
| `SplitValue` | Threshold: points with `x[SplitVar] <= SplitValue` go left |
| `LeftNode` / `RightNode` | Child node indices |
| `NodeWeight` | Number of training observations reaching the node |
| `NodeAve` | Average response in the node (regression forests) |

```python
n_nodes = len(tree["SplitVar"])
n_splits = int((tree["SplitVar"] >= 0).sum())
print(f"{n_nodes} nodes, {n_splits} internal splits")
```

```
41 nodes, 20 internal splits
```

A compact text preview of the tree, in the spirit of R's `get.one.tree()`:

```python
def print_tree_preview(tree, max_lines=14):
    lines = []
    for i, var in enumerate(tree["SplitVar"]):
        if var < 0:
            lines.append(
                f"node {i:>3} | terminal          "
                f"| n = {int(tree['NodeWeight'][i]):>4} "
                f"| mean = {tree['NodeAve'][i]:8.3f}"
            )
        else:
            lines.append(
                f"node {i:>3} | X[{var}] <= {tree['SplitValue'][i]:7.3f} "
                f"| n = {int(tree['NodeWeight'][i]):>4} "
                f"| mean = {tree['NodeAve'][i]:8.3f}"
            )
    for line in lines[:max_lines]:
        print(line)
    if len(lines) > max_lines:
        print("\n... output truncated ...")

print_tree_preview(tree)
```

```
node   0 | X[5] <=   1.500 | n =   80 | mean =    2.832
node   1 | X[5] <=   0.500 | n =   58 | mean =    1.932
node   2 | X[3] <=  -0.248 | n =   42 | mean =    3.813
node   3 | X[1] <=   1.599 | n =   33 | mean =    2.522
node   4 | X[2] <=   0.629 | n =   25 | mean =    4.741
node   5 | X[2] <=  -0.308 | n =   31 | mean =    1.568
...
```

For linear-combination forests, the loadings of every internal split live
in `fit.forest_["SplitLoad"][tree_id]` — one row of `linear_comb`
coefficients per internal node (in node order, matching the internal
`SplitVar` rows):

```python
fit_lc = RLT_reg(n_estimators=50, mtry=3, nsplit=3,
                 linear_comb=3, linear_comb_method="sir",
                 n_jobs=1, random_state=1)
fit_lc.fit(trainX, trainY)

loadings = np.asarray(fit_lc.forest_["SplitLoad"][0])
loadings.shape  # (n_internal_nodes, linear_comb)

np.round(loadings[0], 3)  # coefficients of the first split
```

```
array([0.702, 0.658, 0.272])
```
