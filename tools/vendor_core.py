#!/usr/bin/env python3
"""Vendor the RLT C++ core into src/cpp/, stripping R/Rcpp dependencies.

Idempotent: always rebuilds src/cpp/ from the pristine R package source.
Excluded (by design):
  - quantile/   : out of scope for rlt-python
  - r-interface/: replaced by the pybind11 layer
  - kernels/    : deferred to the kernel milestone (Rcpp-typed signatures)
  - RLT_init.cpp, Makevars*: R build machinery

Transformations:
  - #include <RcppArmadillo.h>  -> #include <armadillo> + "rlt_compat.h"
  - using namespace Rcpp;       -> removed
  - dqrng / boost::random       -> rlt:: vendored RNG (via compat header)
  - Rcpp::warning               -> rlt_warning
  - PARAM_READ_R(Rcpp List)     -> PARAM_READ(rlt::CoreParams const&)
  - RLTcout macro               -> std::cout
"""
import re
import shutil
import sys
from pathlib import Path

SRC = Path("/home/tez/RLT/src")
DST = Path(__file__).resolve().parents[1] / "src" / "cpp"

EXCLUDE_DIRS = {"quantile", "r-interface", "kernels", "include/quantile"}
EXCLUDE_FILES = {"RLT_init.cpp", "Makevars", "Makevars.win", "RcppExports.cpp"}

# ---------------------------------------------------------------- compat header
RLT_COMPAT = r'''// ************************************************************
// rlt-python compatibility header
// Replaces Rcpp/R/dqrng/boost dependencies of the original RLT
// core with self-contained, deterministic implementations.
// ************************************************************
#ifndef RLT_COMPAT_H
#define RLT_COMPAT_H

#include <armadillo>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>

namespace rlt {

// ---------- deterministic RNG: xoshiro256plus ----------
class xoshiro256plus {
  std::uint64_t s[4];

  static std::uint64_t splitmix64(std::uint64_t& x) {
    std::uint64_t z = (x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
  }

  static std::uint64_t rotl(const std::uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
  }

 public:
  xoshiro256plus() : s{1, 0, 0, 0} {}
  explicit xoshiro256plus(std::uint64_t seed) {
    std::uint64_t sm = seed;
    for (auto& v : s) v = splitmix64(sm);
    if (!(s[0] | s[1] | s[2] | s[3])) s[0] = 1;  // never all-zero
  }

  std::uint64_t next() {
    const std::uint64_t result = s[0] + s[3];
    const std::uint64_t t = s[1] << 17;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = rotl(s[3], 45);
    return result;
  }

  double uniform01() { return static_cast<double>(next() >> 11) * 0x1.0p-53; }
};

// ---------- Lemire debiased bounded integers, [min, max] ----------
template <typename T>
class uniform_int_distribution {
 public:
  using result_type = T;

  uniform_int_distribution(T min_v, T max_v)
      : min_(min_v),
        range_(static_cast<std::uint64_t>(max_v) -
               static_cast<std::uint64_t>(min_v) + 1ULL) {}

  template <typename RNG>
  T operator()(RNG& rng) const {
    const std::uint64_t thresh = (~range_ + 1ULL) % range_;  // 2^64 mod range
    std::uint64_t x;
    do {
      x = rng.next();
    } while (x < thresh);
    return min_ + static_cast<T>(x % range_);
  }

 private:
  T min_;
  std::uint64_t range_;
};

// ---------- uniform reals [min, max] ----------
template <typename T>
class uniform_real_distribution {
 public:
  using result_type = T;

  uniform_real_distribution(T min_v, T max_v)
      : min_(min_v), span_(max_v - min_v) {}

  template <typename RNG>
  T operator()(RNG& rng) const {
    return min_ + static_cast<T>(rng.uniform01()) * span_;
  }

 private:
  T min_;
  T span_;
};

// ---------- plain parameter struct (replaces the Rcpp List) ----------
struct CoreParams {
  // main parameters
  size_t n = 0;
  size_t p = 0;
  size_t ntrees = 1;
  size_t mtry = 1;
  size_t nmin = 1;
  size_t nsplit = 1;
  bool replacement = false;
  double resample_prob = 0.8;
  bool use_obs_w = false;
  bool use_var_prob = false;
  size_t importance = 0;
  bool reinforcement = false;
  // other control parameters
  bool obs_track = false;
  size_t var_mode = 0;
  size_t linear_comb = 1;
  double alpha = 0;
  size_t split_rule = 1;
  size_t linear_comb_method = 1;
  // embedded (reinforcement) forest parameters
  size_t embed_ntrees = 0;
  double embed_mtry = 0;
  size_t embed_nmin = 0;
  size_t embed_nsplit = 0;
  bool embed_replacement = false;
  double embed_resample_prob = 0;
  double embed_mute = 0;
  size_t embed_protect = 0;
  double embed_threshold = 0.25;
  // system
  size_t ncores = 1;
  size_t verbose = 0;
  size_t seed = 1;
};

}  // namespace rlt

// warning shim: routes through std::cerr (Python layer maps to warnings)
inline void rlt_warning(const std::string& msg) {
  std::cerr << "RLT warning: " << msg << std::endl;
}

// printf-style shim for the couple of Rprintf() sites in the core
#include <cstdarg>
#include <cstdio>
inline void Rprintf(const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  std::vfprintf(stderr, fmt, ap);
  va_end(ap);
  std::fprintf(stderr, "\n");
}

#endif  // RLT_COMPAT_H
'''

PARAM_READ_NEW = r'''  void PARAM_READ(const rlt::CoreParams& q) {
    N = q.n;                    P = q.p;
    ntrees = q.ntrees;          mtry = q.mtry;
    nmin = q.nmin;              nsplit = q.nsplit;
    replacement = q.replacement;
    resample_prob = q.resample_prob;
    useobsweight = q.use_obs_w; usevarprob = q.use_var_prob;
    importance = q.importance;  reinforcement = q.reinforcement;
    obs_track = q.obs_track;    var_mode = q.var_mode;
    linear_comb = q.linear_comb; alpha = q.alpha;
    split_rule = q.split_rule;
    linear_comb_method = q.linear_comb_method;
    embed_ntrees = q.embed_ntrees;        embed_mtry = q.embed_mtry;
    embed_nmin = q.embed_nmin;            embed_nsplit = q.embed_nsplit;
    embed_replacement = q.embed_replacement;
    embed_resample_prob = q.embed_resample_prob;
    embed_mute = q.embed_mute;            embed_protect = q.embed_protect;
    embed_threshold = q.embed_threshold;
    ncores = q.ncores;          verbose = q.verbose;
    seed = q.seed;
  };'''

RLT_H = r'''//    ----------------------------------------------------------------
//
//    Reinforcement Learning Trees (RLT) - python port main header
//    Core + regression + classification + survival
//    (quantile module intentionally excluded from rlt-python)
//
//    ----------------------------------------------------------------

#ifndef RLT_H
#define RLT_H

#include "rlt_compat.h"

// Core infrastructure
#include "include/core/Utility.h"
#include "include/core/Tree_Definition.h"
#include "include/core/Tree_Function.h"
#include "include/core/Stat_Function.h"
#include "include/core/Variance_IJ_Jack.h"

// Regression
#include "include/regression/Reg_Uni_Definition.h"
#include "include/regression/Reg_Uni_Function.h"

// Classification
#include "include/classification/Cla_Uni_Definition.h"
#include "include/classification/Cla_Uni_Function.h"

// Survival
#include "include/survival/Surv_Uni_Definition.h"
#include "include/survival/Surv_Uni_Function.h"

#endif  // RLT_H
'''


def splice_method(text: str, header: str, replacement: str) -> str:
    """Replace the brace-balanced block starting at `header` (a signature
    prefix) with `replacement` (which must itself be balanced)."""
    start = text.index(header)
    brace_open = text.index("{", start)
    depth, i = 0, brace_open
    while True:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                break
        i += 1
    end = i + 1
    # consume trailing semicolon if present
    while end < len(text) and text[end] in " \t":
        end += 1
    if end < len(text) and text[end] == ";":
        end += 1
    return text[:start] + replacement + text[end:]


def transform(text: str, path: Path) -> str:
    # 1. RcppArmadillo -> armadillo + compat
    text = re.sub(r'#\s*include\s*<RcppArmadillo\.h>',
                  '#include <armadillo>\n#include "rlt_compat.h"', text)
    # 2. drop R namespace + dqrng/boost includes
    text = re.sub(r'^\s*using\s+namespace\s+Rcpp\s*;\s*\n', '', text,
                  flags=re.M)
    text = re.sub(r'#\s*include\s*<dqrng[^>]*>\s*\n?', '', text)
    text = re.sub(r'#\s*include\s*<xoshiro[^>]*>\s*\n?', '', text)
    text = re.sub(r'#\s*include\s*<boost[^>]*>\s*\n?', '', text)
    # 3. RNG swaps
    text = text.replace("dqrng::xoshiro256plus", "rlt::xoshiro256plus")
    text = text.replace("boost::random::uniform_int_distribution",
                        "rlt::uniform_int_distribution")
    text = text.replace("boost::random::uniform_real_distribution",
                        "rlt::uniform_real_distribution")
    # 4. warnings
    text = text.replace("Rcpp::warning", "rlt_warning")
    # 4b. R logical literals
    text = re.sub(r'\bTRUE\b', 'true', text)
    text = re.sub(r'\bFALSE\b', 'false', text)
    # 4c. drop R-facing entry-point declarations (List XxxForestFit(...);)
    #     their implementations live in the excluded r-interface layer;
    #     the pybind11 layer declares its own typed entry points.
    text = re.sub(r'\nList\s+\w+\([^;]*?\);\n', '\n', text, flags=re.S)
    # 5. RLTcout -> std::cout (defined once in Utility.h)
    text = text.replace("#define RLTcout Rcout", "#define RLTcout std::cout")
    # 6. Rcpp::as / Rf_length sites (PARAM_READ_R) — spliced below
    return text


def main() -> int:
    if not SRC.is_dir():
        print(f"source not found: {SRC}", file=sys.stderr)
        return 1
    if DST.exists():
        shutil.rmtree(DST)
    DST.mkdir(parents=True)

    copied, skipped = [], []
    for p in sorted(SRC.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(SRC).as_posix()
        if any(rel.startswith(d) or rel == d.rstrip("/") for d in EXCLUDE_DIRS):
            skipped.append(rel)
            continue
        if p.name in EXCLUDE_FILES:
            skipped.append(rel)
            continue
        if p.suffix not in {".h", ".cpp"}:
            continue
        text = p.read_text()
        text = transform(text, p)
        out = DST / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
        copied.append(rel)

    # --- Utility.h specials: PARAM_READ_R splice ---
    utility = DST / "include/core/Utility.h"
    text = utility.read_text()
    text = splice_method(text, "void PARAM_READ_R(List& param)", PARAM_READ_NEW)
    utility.write_text(text)

    # --- top-level headers ---
    (DST / "rlt_compat.h").write_text(RLT_COMPAT)
    (DST / "RLT.h").write_text(RLT_H)

    print(f"vendored {len(copied)} files -> {DST}")
    print(f"skipped {len(skipped)} files (by design)")

    # --- residual audit: anything still smelling of R/Rcpp ---
    bad = re.compile(r"Rcpp::|RcppExport|\bRcout\b|\bRf_|\bSEXP\b|"
                     r"RcppArmadillo|\bdqrng\b|boost::random|"
                     r"R Inside|R_inside")
    residuals = {}
    for p in sorted(DST.rglob("*")):
        if p.is_file() and p.suffix in {".h", ".cpp"}:
            for n, line in enumerate(p.read_text().splitlines(), 1):
                if bad.search(line):
                    residuals.setdefault(p.relative_to(DST).as_posix(),
                                        []).append(f"{n}: {line.strip()}")
    if residuals:
        print("\n=== RESIDUAL R/Rcpp REFERENCES ===")
        for f, lines in residuals.items():
            print(f"--- {f}")
            for ln in lines[:12]:
                print("   ", ln)
    else:
        print("\nno residual R/Rcpp references — clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
