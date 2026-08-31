"""Probe classification + survival APIs."""
import time
import numpy as np
import rlt

# ---------- classification ----------
rng = np.random.default_rng(1)
trainn, testn, p = 80, 20, 10
n = trainn + testn
X1 = rng.normal(size=(n, p // 2))
X2 = (rng.uniform(size=(n, p // 2)) * 10).astype(int)  # integers 0-9
X = np.hstack([X1, X2])

def sigmoid(x):
    return np.exp(x) / (1 + np.exp(x))

prob = sigmoid(-0.5 + 2 * X[:, 0])
y = rng.binomial(1, prob)

trainX, trainY = X[:trainn], y[:trainn]
testX, testY = X[trainn:], y[trainn:]

fit = rlt.RLT_cla(
    n_estimators=200, mtry=p // 2, min_samples_leaf=5,
    resample_prob=0.8, resample_replace=True, nsplit=0,
    importance="permute", n_jobs=1, random_state=1,
)
fit.fit(trainX, trainY)
print("cla classes_:", fit.classes_)
print("cla oob_error_:", fit.oob_error_)
proba = fit.predict_proba(testX)
print("proba shape:", proba.shape, "rowsum:", proba[0].sum())
labels = fit.predict(testX)
print("labels:", labels[:5], "dtype:", labels.dtype)
train_acc = float(np.mean(fit.oob_class_ == trainY))
test_acc = float(np.mean(labels == testY))
print("train acc (oob):", train_acc, "test acc:", test_acc)
print("varimp:", np.round(fit.feature_importances_, 2))

# string labels
fit2 = rlt.RLT_cla(n_estimators=50, random_state=0)
y_str = np.array(["neg", "pos"])[y]
fit2.fit(X, y_str)
print("string classes_:", fit2.classes_, fit2.predict(X[:3]))

# ---------- survival ----------
rng = np.random.default_rng(42)
n, p = 200, 5
X = rng.normal(size=(n, p))
beta = np.array([0.8, 0.5, 0, 0, 0])
hazard = np.exp(X @ beta)
surv_time = rng.exponential(1.0 / hazard)
censor_time = rng.uniform(0, 3, n)
y_time = np.minimum(surv_time, censor_time)
event = (surv_time <= censor_time).astype(int)
print("events:", event.sum(), "/", n)

ys = np.empty(n, dtype=[("event", "?"), ("time", "<f8")])
ys["event"] = event.astype(bool)
ys["time"] = y_time

t0 = time.time()
fit = rlt.RLT_surv(n_estimators=100, min_samples_leaf=5, n_jobs=1, random_state=42)
fit.fit(X, ys)
print("surv fit %.2fs" % (time.time() - t0))
print("oob_error_:", fit.oob_error_)
S = fit.predict_survival_function(X[:5])
print("S shape:", S.shape, "timepoints:", fit.timepoints_.shape, "first tp:", fit.timepoints_[:3])
print("S row0 head:", np.round(S[0, :5], 3), "S last:", round(float(S[0, -1]), 3))
chf = fit.predict_cumulative_hazard_function(X[:5])
print("CHF shape:", chf.shape)
print("c-index score:", fit.score(X, ys))
print("predict_risk:", fit.predict_risk(X[:3]))

# split rules
for rule in ["logrank", "suplogrank", "coxgrad"]:
    f = rlt.RLT_surv(n_estimators=100, split_rule=rule, n_jobs=1, random_state=42)
    f.fit(X, ys)
    print(rule, "oob_error_:", round(f.oob_error_, 5))

# obs weights with coxgrad
w = rng.uniform(size=n)
fw = rlt.RLT_surv(n_estimators=100, split_rule="coxgrad", n_jobs=1, random_state=1)
fw.fit(X, ys, sample_weight=w)
print("weighted coxgrad ok, oob:", round(fw.oob_error_, 5))

# LC
flc = rlt.RLT_surv(n_estimators=100, split_rule="logrank", linear_comb=3,
                   linear_comb_method="coxph", n_jobs=1, random_state=1)
flc.fit(X, ys)
print("surv lc ok, oob:", round(flc.oob_error_, 5))
print("surv lc pred:", flc.predict_survival_function(X[:5]).shape)

# importance
fi = rlt.RLT_surv(n_estimators=100, importance="permute", n_jobs=1, random_state=1)
fi.fit(X, ys)
print("surv varimp:", np.round(fi.feature_importances_, 2))

# get one tree
tree = fit.get_one_tree(0)
print("surv tree keys:", list(tree.keys()))
sv = tree["SplitVar"]
print("surv tree nodes:", len(sv))

# variance + bands
t0 = time.time()
fvar = rlt.RLT_surv(n_estimators=1000, var_mode="matched", n_jobs=4, random_state=42)
fvar.fit(X, ys)
print("var fit %.1fs" % (time.time() - t0))
Sv, cov = fvar.predict_var(X[:3])
print("Sv:", Sv.shape, "cov:", cov.shape)
t0 = time.time()
band = rlt.get_surv_band(fvar, X[:2], i=0, alpha=0.05, approach="smoothed", nsim=2000)
print("band %.1fs" % (time.time() - t0))
print("band keys:", sorted(band.keys()))
print("lower/upper:", band["lower"].shape, band["upper"].shape)
print("timepoints:", band["timepoints"].shape)
print("cover:", float(np.mean((band["lower"][0] <= Sv[0]) & (Sv[0] <= band["upper"][0]))))

band2 = rlt.get_surv_band(fvar, X[:2], i=1, alpha=0.05, approach="naive", nsim=2000)
print("naive band:", band2["lower"].shape)

# band_grid_size
S3, cov3 = fvar.predict_var(X[:3], band_grid_size=50)
print("reduced grid:", S3.shape, cov3.shape)
