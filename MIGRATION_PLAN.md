# RLT → Python Migration Plan

**Source:** R package `RLT` v6.0.2 (`/home/tez/RLT`, CRAN `teazrq/RLT`)
**Target:** Python package `rlt-forests` (this repo)

## Status

| Milestone | State |
|---|---|
| M0 scaffolding, build, git | ✅ done |
| M1 core lift (Rcpp stripped, compiles) | ✅ done — 40 TUs, plain g++/Armadillo/OpenMP |
| M2 regression vertical slice | ✅ done |
| M3 classification + survival | ✅ done |
| M4 reinforcement / linear-comb forests | ⬜ next (C++ already compiled in; needs estimator surface + Comb fit/pred bindings) |
| M5 importance, kernels, get_one_tree polish, survival bands (U-stat + tensor-product spline in C++) | ⬜ partial (importance + get_one_tree done; kernels + bands pending) |
| M6 docs, CI, PyPI | ⬜ pending |

## Locked design decisions (user, 2026-08-30)

1. **RNG: numpy seeding.** `random_state` (int/RandomState) → master seed via
   `check_random_state`; per-tree seeds derived inside C++ (vendored
   xoshiro256plus + Lemire bounded ints — deterministic across platforms).
   No attempt at bit-compatibility with the R package.
2. **U-statistics: heavy computation in C++.** The IJ/jackknife cores were
   already Armadillo C++ and are ported. The survival confidence-band
   machinery needs a **tensor-product spline** — implement carefully, in C++,
   as its own milestone (M5).
3. **Full sklearn compatibility from day one.** `RLTRegressor`,
   `RLTClassifier`, `RLTSurvivalForest` on `BaseEstimator`; `fit/predict/
   predict_proba/feature_importances_/get_params/set_params`, structured-array
   survival `y` (`event`/`time`, scikit-survival convention), `score()` =
   Harrell's c-index. Verified with sklearn `clone`.
4. **Scope: regression, classification, survival only.** Quantile module
   excluded from the vendored core.

## Architecture

```
src/cpp/        vendored compute core (tools/vendor_core.py regenerates
                idempotently from /home/tez/RLT/src; Rcpp/R/RNG stripped,
                rlt_compat.h shims: xoshiro256plus, Lemire uniforms,
                CoreParams struct, Rprintf/rlt_warning)
src/bindings/   pybind11 layer (_core module): typed fit/pred entry points,
                numpy <-> armadillo converters, cindex, matched ObsTrack gen
rlt/            user API: estimators.py (sklearn), _params.py
tests/          pytest (9 passing) + differential-vs-R scripts
```

## Verification so far

- 9/9 pytest passing (fit/predict all models, reproducibility, var modes,
  input validation, string classes)
- Differential vs R package (same data, same hyper-params, 500 trees):
  reg pred corr 0.90 (MC noise), survival curves corr 0.995, OOB errors
  match (reg 0.99 vs 1.00; cla 0.148 vs 0.160; surv 0.448 vs 0.436)
- 2000-tree best-split run: OOB reg error R 0.9987 vs py 1.0044 — same
  estimator to Monte-Carlo noise

## Known pitfalls (fixed — do not regress)

- **ncat convention: 1 = continuous, 0 unused, >1 = categorical.** Passing
  zeros sent every split into the categorical path (OOB `goright[x]` reads,
  silent garbage). Estimators default `ncat = ones(p)`.
- Armadillo aux-memory constructor is column-major-only; never view C-order
  numpy buffers through it (explicit copies in bindings).
- Link with `-DARMA_DONT_USE_WRAPPER -llapack -lblas` (no armadillo wrapper
  lib).
- `cindex_d/cindex_i` lived in the excluded r-interface but are called by the
  survival core — reimplemented in vendored `core/Stat_Function.cpp`.
- RLT's `cindex(pred=risk)`: higher risk = higher event probability.

## Next steps (M4)

1. Bind `RegUniCombForestFit/Pred`, `ClaUniComb…`, `SurvUniComb…` (SplitLoad
   matrices in/out).
2. Estimator support: `linear_comb > 1` routing + `reinforcement=True`
   embedded models (C++ already builds — only the binding surface is missing).
3. Then M5: kernels (forest similarity), `get.surv.band` port (tensor-product
   spline basis + U-statistic covariance → C++).
4. CI (GH Actions: build matrix linux/mac/win) before PyPI.
