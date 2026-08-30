// Minimal repro: call split_id directly with x=[1..6], value=3.5
#include "RLT.h"
#include <cstdio>
using namespace arma;

int main() {
  arma::mat X(6, 1, fill::zeros);
  for (size_t i = 0; i < 6; i++) X(i, 0) = (double)(i + 1);
  arma::vec x_copy = X.unsafe_col(0);

  uvec obs_id = linspace<uvec>(0, 5, 6);
  uvec left_id(6);

  split_id(x_copy, 3.5, left_id, obs_id);
  std::printf("copy  -> left=%zu :", (size_t)left_id.n_elem);
  for (auto v : left_id) std::printf(" %zu", (size_t)v);
  std::printf(" | right=");
  for (auto v : obs_id) std::printf(" %zu", (size_t)v);
  std::printf("\n");

  // now via subview expression directly
  uvec obs_id2 = linspace<uvec>(0, 5, 6);
  uvec left_id2(6);
  split_id(conv_to<vec>::from(X.unsafe_col(0)), 3.5, left_id2, obs_id2);
  std::printf("conv  -> left=%zu :", (size_t)left_id2.n_elem);
  for (auto v : left_id2) std::printf(" %zu", (size_t)v);
  std::printf(" | right=");
  for (auto v : obs_id2) std::printf(" %zu", (size_t)v);
  std::printf("\n");
  return 0;
}
