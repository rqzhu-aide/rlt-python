# RLT Classification Tutorial

This page shows how to fit and predict a classification model with `rlt` —
the Python port of the RLT R package's classification vignette.

## Prerequisites

Install `rlt` (`pip install rlt-forests`), then import it:

```python
from rlt import RLT_cla
```

## Data Preparation

We generate continuous and categorical predictors with a binary outcome.

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
X2 = (rng.uniform(size=(n, p // 2)) * 10).astype(int)  # integers 0-9
X = np.hstack([X1, X2])

# Binary outcome via a simple logistic signal
def sigmoid(x):
    return np.exp(x) / (1 + np.exp(x))

prob = sigmoid(-0.5 + 2 * X[:, 0])  # signal in the first continuous feature
y = rng.binomial(1, p=prob)

# Train / test split
trainX, trainY = X[:trainn], y[:trainn]
testX, testY = X[trainn:], y[trainn:]
```

## Fit a classification model

```python
from rlt import RLT_cla

ntrees = 200
nmin = 5
mtry = p // 2
samplereplace = True
sampleprob = 0.80
rule = "best"
nsplit = 0 if rule == "best" else 3
importance = "permute"

fit = RLT_cla(
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

## Predict

```python
# Predicted class labels (original label type: 0/1 here)
labels = fit.predict(testX)

# Class-membership probabilities, shape (n, n_classes)
proba = fit.predict_proba(testX)

# OOB class labels and error from the fit
train_pred = fit.oob_class_
```

## Evaluate

```python
acc_train = float(np.mean(train_pred == trainY))
acc_test = float(np.mean(labels == testY))

# A compact summary
print(f"Train accuracy: {acc_train:.4f}")
print(f"Test accuracy:  {acc_test:.4f}")
```

```
Train accuracy: 0.7375
Test accuracy:  0.8000
```

## Inspect the fitted object

```python
print(fit)
```

```
RLT_cla(importance='permute', mtry=5, n_jobs=1, n_estimators=200,
        nsplit=0, random_state=1, resample_prob=0.8)
```

Fitted attributes worth knowing:

```python
fit.classes_          # sorted unique labels seen at fit time
fit.n_features_in_    # number of predictors
fit.oob_error_        # out-of-bag misclassification rate
fit.oob_class_        # (n,) OOB predicted classes (integer codes)
fit.oob_prob_         # (n, n_classes) OOB class probabilities
fit.feature_importances_  # (p,) variable importance
```

String labels work too — `RLT_cla` stores `classes_` and `predict` returns
the original label type:

```python
y_str = np.array(["neg", "pos"])[y]
fit2 = RLT_cla(n_estimators=200, random_state=1)
fit2.fit(X, y_str)
fit2.classes_        # array(['neg', 'pos'], dtype='<U3')
fit2.predict(X[:3])  # e.g. array(['pos', 'neg', 'neg'], dtype='<U3')
```
