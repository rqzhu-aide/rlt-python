// ************************************************************
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

// ---------- boost::random-exact distributions ----------
// Bit-exact replicas of boost::random::uniform_int_distribution and
// uniform_real_distribution as consumed by the original RLT core with a
// 64-bit integral engine (dqrng xoshiro256plus: min=0, max=2^64-1).
// This is what makes same-seed fits identical to the R package.

template <typename T>
class uniform_int_distribution {
 public:
  using result_type = T;

  uniform_int_distribution(T min_v, T max_v)
      : min_(min_v),
        range_(static_cast<std::uint64_t>(max_v) -
               static_cast<std::uint64_t>(min_v)) {}

  template <typename RNG>
  T operator()(RNG& rng) const {
    // boost::random::detail::generate_uniform_int, integral-engine branch
    const std::uint64_t brange = ~std::uint64_t(0);  // eng.max - eng.min
    if (range_ == 0) return min_;  // boost draws NOTHING here
    std::uint64_t result;
    if (brange == range_) {
      result = rng.next();
    } else {
      // brange == numeric_limits<base_unsigned>::max() branch:
      std::uint64_t bucket_size = brange / (range_ + 1);
      if (brange % (range_ + 1) == range_) ++bucket_size;
      for (;;) {
        result = rng.next() / bucket_size;
        if (result <= range_) break;
      }
    }
    return static_cast<T>(result + min_);
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

  uniform_real_distribution(T min_v, T max_v) : min_(min_v), max_(max_v) {}

  template <typename RNG>
  T operator()(RNG& rng) const {
    // boost::random::detail::generate_uniform_real, integral-engine branch:
    // numerator / (double)(eng.max - eng.min + 1) * span + min, retry if
    // the rounding hits the upper bound.
    const T divisor = static_cast<T>(~std::uint64_t(0)) + 1.0;  // 2^64
    for (;;) {
      T numerator = static_cast<T>(rng.next());
      T result = numerator / divisor * (max_ - min_) + min_;
      if (result < max_) return result;
    }
  }

 private:
  T min_, max_;
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
