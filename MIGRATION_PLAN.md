# RLT → Python Migration Plan

**Source:** R package `RLT` v6.0.2 (`/home/tez/RLT`, CRAN `teazrq/RLT`)
**Target:** Python package `rlt` (this repo)

## 1. What RLT is

Random forest with extensions for **regression, classification, survival, and quantile** analysis:

- Parallel computing with OpenMP, reproducibility with random seeds
- Variance and confidence band estimations using U-statistics (jackknife/IJ)
- Embedded models for selecting splitting variables and constructing linear-combination splits ("reinforcement")
- Permutation and distribution-based variable importance
- Observation and variable weights; subject tracking across trees (kkit files / one-hot tree kernels)

**Structure (~19k lines C++ + ~3.5k lines R):**

| Layer | Contents | R/Rcpp coupling |
|---|---|---|
| `src/core`, `src/kernels`, `src/{regression,classification,survival,quantile}` | Compute: tree building, split finding, scoring, forests | Armadillo types in data structures; ~5 stray `Rcpp::warning`/`Rcpp::as` calls; dqrng (xoshiro256plus) + `boost::random` distributions for RNG |
| `src/r-interface` (9 files + RcppExports) | SEXP marshalling of 25 exported functions | Full Rcpp — **replaced wholesale** by a pybind11 layer |
| `R/*.r` (15 files) | User API: `RLT()`, `predict()`, `importance()`, forest kernels, checks | Pure R — **re-implemented** as the Python API |

Key facts from recon:

- RNG: master `Rand rng(seed)` draws per-tree seeds (`rand_uvec`), each tree builds with `Rand rngl(seed_vec(nt))`. dqrng's xoshiro256plus + boost distributions. **Cross-language seed reproducibility (R ↔ Python identical trees) is possible only if we port the RNG bit-for-bit** (xoshiro256plus is a simple public algorithm; boost distributions are deterministic given the engine).
- `Param` class holds all tuning parameters, set from an R list at the interface layer.
- Trees are stored as flat arrays (`SplitVar`, `SplitValue`, `LeftNode`, `RightNode`, `NodeWeight`, …) — maps naturally to numpy arrays / a dataclass.
- Linear-combination (reinforcement) forests add `SplitLoad` and use embedded ridge/logistic fits via Armadillo `solve`.
- The R layer holds real logic: input checking, resample logic (`resample.preset`, `gen_ms_obs_track_mat`), importance computation, survival band estimation (U-statistic machinery in `cindex.r`, `get.surv.band.r`).

## 2. Strategy (recommended)

**Keep the C++ core; replace the binding layer; rewrite the user-facing API in Python.**

- **`src/cpp/`** — vendored compute core, lifted from RLT with a thin cleanup pass:
  - Remove `#include <RcppArmadillo.h>` → `#include <armadillo>` (+ a small `rlt_compat.h` shimming the ~5 stray `Rcpp::*` uses and the RNG objects).
  - RNG: vendor xoshiro256plus (dqrng's implementation is tiny) + the boost distribution call sites stay boost (header-only) OR get a deterministic shim. Decision point: strict bit-compatibility with R vs. self-consistent Python-only seeding. **Recommendation: bit-compatible**, it makes R↔Python differential testing trivial ("same seed → same forest"), which is the single best correctness tool for the whole migration.
- **`src/bindings/`** — new pybind11 module exposing the same 25-entry surface as typed C++ functions (matrices in, structured arrays out). No Rcpp anywhere.
- **`rlt/`** (Python) — user API mirroring the R package:
  - `RLT(x, y, censor=None, model=..., ...)` returning `RegForest` / `ClaForest` / `SurvForest` / `QuanForest` estimator objects with `.fit`-compatible construction, `predict()`, `importance()`, `get_one_tree()`, kernel functions.
  - **sklearn-compatible facade** (optional but valuable): `fit`/`predict`/`feature_importances_`, so `rlt` slots into cross-validation pipelines.
- **`tests/`** — pytest suite, seeded, mirroring `tests/testthat/`; plus **differential tests vs the R package** (run both, compare forests/predictions to tolerance).

Packaging: `scikit-build-core` or plain `meson`/setuptools+pybind11, building `rlt._core` from `src/cpp`. CI on GitHub Actions (ubuntu + macOS + Windows).

## 3. Milestones

1. **M0 — Scaffolding** *(this week)*: package layout, build system compiling the untouched core with plain Armadillo, CI skeleton.
2. **M1 — Core lift**: strip Rcpp from core (compat header + RNG vendoring), compile, unit-test RNG bit-compatibility against R.
3. **M2 — Regression vertical slice**: pybind11 bindings for `RegUniForestFit/Pred` + Python `RLT()` for regression only. End-to-end test vs R output.
4. **M3 — Classification + quantile + survival**: remaining fit/pred bindings, `Param` surface complete.
5. **M4 — Linear combination / reinforcement forests** (UniComb variants + embedded models).
6. **M5 — R-layer features**: importance, kernels, `get_one_tree`, survival bands (U-statistics — port or re-derive in numpy/scipy).
7. **M6 — sklearn facade, docs, PyPI release.**

## 4. Open questions

1. **RNG fidelity**: strict bit-compatibility with R (vendored xoshiro + boost) vs clean Python seeding (`numpy.random` per tree)? Affects M1 scope.
2. **U-statistic variance machinery** (`get.surv.band.r`, IJ/jackknife): port the R code to numpy, or also lift the C++ in `core/`?
3. **sklearn facade** now (shapes the API from M2) or later (M6)?
4. **Scope**: full feature parity including quantile forests + kernels, or regression/classification/survival first and quantile on demand?
5. **PyPI name**: `rlt` is taken-ish on PyPI (check); candidates: `rlt-forests`, `reinforcement-learning-trees`, `rltrees`.
