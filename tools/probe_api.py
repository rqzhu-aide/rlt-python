"""Probe the rlt Python API to ground the doc examples."""
import time
import numpy as np
import rlt

print("version:", rlt.__version__)

# ---------- regression ----------
rng = np.random.default_rng(1)
trainn, testn, p = 80, 20, 10
n = trainn + testn
X1 = rng.normal(size=(n, p // 2))
X2 = (rng.uniform(size=(n, p // 2)) * 3).astype(int)  # integers 0,1,2
X = np.hstack([X1, X2])
y = 1 + X[:, 1:6].sum(axis=1) + 2 * np.isin(X[:, p // 2], [1, 2]) + rng.normal(size=n)

trainX, trainY = X[:trainn], y[:trainn]
testX, testY = X[trainn:], y[trainn:]

t0 = time.time()
fit = rlt.RLT_reg(
    n_estimators=200, mtry=p // 2, min_samples_leaf=5,
    resample_prob=0.8, resample_replace=True, nsplit=0,
    importance="permute", n_jobs=1, random_state=1,
)
fit.fit(trainX, trainY)
print("reg fit %.1fs" % (time.time() - t0))
pred = fit.predict(testX)
train_pred = fit.oob_prediction_
print("oob_error_:", fit.oob_error_)
print("train mse (oob pred):", float(np.mean((train_pred - trainY) ** 2)))
print("test mse:", float(np.mean((pred - testY) ** 2)))
print("varimp shape:", fit.feature_importances_.shape)

# get_one_tree
tree = fit.get_one_tree(0)
print("tree keys:", list(tree.keys()))
sv = tree["SplitVar"]
print("n nodes:", len(sv), "n splits:", int((sv >= 0).sum()))

# predict_var
fitv = rlt.RLT_reg(n_estimators=200, var_mode="matched", random_state=1)
fitv.fit(trainX, trainY)
pv, var = fitv.predict_var(testX)
print("predict_var shapes:", pv.shape, var.shape, "NaNs in var:", int(np.isnan(var).sum()), "/", var.shape[0])

# forest kernel
fitk = rlt.RLT_reg(n_estimators=300, mtry=p, min_samples_leaf=5,
                   resample_prob=0.8, resample_replace=True,
                   importance="permute", resample_track=True,
                   n_jobs=1, random_state=1)
fitk.fit(X, y)
newX = np.full((1, p), 0.5)
newX[0, 1] = 0.3
K = fitk.forest_kernel(newX, X)
print("kernel cross:", K.shape, K.dtype)
Kt = fitk.forest_kernel(newX, X, vs_train=True)
print("kernel train:", Kt.shape)
Ks = fitk.forest_kernel(X[:5])
print("kernel self:", Ks.shape)

# LC splits
t0 = time.time()
fit_lc = rlt.RLT_reg(n_estimators=200, min_samples_leaf=5, mtry=3,
                     nsplit=3, linear_comb=3, linear_comb_method="sir",
                     n_jobs=1, random_state=3)
fit_lc.fit(trainX, trainY)
print("lc fit %.1fs" % (time.time() - t0))
print("lc oob:", fit_lc.oob_error_)

# reinforcement
t0 = time.time()
fit_r = rlt.RLT_reg(n_estimators=100, reinforcement=True, n_jobs=1, random_state=1)
fit_r.fit(trainX, trainY)
print("reinforcement fit %.1fs, oob %.4f" % (time.time() - t0, fit_r.oob_error_))
