// Debug harness: fit a 6-obs regression forest and dump tree internals.
#include "RLT.h"
#include <cstdio>

using namespace arma;

int main() {
  size_t n = 6, p = 1;
  arma::mat X(n, p, fill::zeros);
  arma::vec Y(n);
  for (size_t i = 0; i < n; i++) { X(i, 0) = (double)(i + 1); Y(i) = (double)(i + 1); }
  arma::uvec Ncat(p, fill::zeros);
  arma::vec obsw(n, fill::value(1.0 / n));
  arma::vec varp(p, fill::value(1.0));

  PARAM_GLOBAL Param;
  Param.N = n; Param.P = p;
  Param.ntrees = 1; Param.mtry = 1; Param.nmin = 3; Param.nsplit = 0;
  Param.replacement = false; Param.resample_prob = 1.0;
  Param.importance = 0; Param.reinforcement = false;
  Param.obs_track = false; Param.var_mode = 0;
  Param.linear_comb = 1; Param.alpha = 0;
  Param.split_rule = 1; Param.linear_comb_method = 1;
  Param.ncores = 1; Param.verbose = 0; Param.seed = 1;

  RLT_REG_DATA REG_DATA(X, Y, Ncat, obsw, varp);

  arma::field<arma::ivec> SplitVar(1);
  arma::field<arma::vec> SplitValue(1);
  arma::field<arma::uvec> LeftNode(1);
  arma::field<arma::uvec> RightNode(1);
  arma::field<arma::vec> NodeWeight(1);
  arma::field<arma::vec> NodeAve(1);

  Reg_Uni_Forest_Class FOREST(SplitVar, SplitValue, LeftNode, RightNode,
                              NodeWeight, NodeAve);

  uvec obs_id = linspace<uvec>(0, n - 1, n);
  uvec var_id = linspace<uvec>(0, p - 1, p);
  arma::imat ObsTrack;  // empty = not pre-defined (C++ fills it)
  vec Prediction; uvec oob_count; vec VarImp; vec VarVI;
  bool do_prediction = false;

  Reg_Uni_Forest_Build(REG_DATA, FOREST, Param, obs_id, var_id, ObsTrack,
                       do_prediction, Prediction, oob_count, VarImp, VarVI);

  ivec sv = SplitVar(0);
  vec svc = SplitValue(0);
  uvec ln = LeftNode(0);
  uvec rn = RightNode(0);
  vec na = NodeAve(0);
  vec nw = NodeWeight(0);

  std::printf("SplitVar:");   for (auto v : sv)  std::printf(" %d", (int)v); std::printf("\n");
  std::printf("SplitValue:"); for (double v : svc) std::printf(" %.3f", v); std::printf("\n");
  std::printf("LeftNode:");  for (auto v : ln)  std::printf(" %llu", (unsigned long long)v); std::printf("\n");
  std::printf("RightNode:"); for (auto v : rn)  std::printf(" %llu", (unsigned long long)v); std::printf("\n");
  std::printf("NodeAve:");   for (double v : na) std::printf(" %.3f", v); std::printf("\n");
  std::printf("NodeWeight:");for (double v : nw) std::printf(" %.1f", v); std::printf("\n");
  std::printf("expected: cut 3.5, leaves 2.0 and 5.0\n");
  return 0;
}
