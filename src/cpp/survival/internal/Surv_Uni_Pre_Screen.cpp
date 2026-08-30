//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Survival
//  **********************************

// my header file
# include "RLT.h"
using namespace arma;

// Embedded RF VI Screening Method for Survival
vec Surv_Uni_Embed_Pre_Screen(const RLT_SURV_DATA& SURV_DATA,
                              const PARAM_GLOBAL& Param,
                              const uvec& obs_id,
                              const uvec& var_id,
                              Rand& rngl)
{
  // set embedded model parameters
  PARAM_GLOBAL Embed_Param;

  Embed_Param.N = obs_id.n_elem;
  Embed_Param.P = var_id.n_elem;
  Embed_Param.ntrees = Param.embed_ntrees;

  // mtry can be < 1 or >= 1
  if (Param.embed_mtry >= 1)
    Embed_Param.mtry = (size_t) Param.embed_mtry;
  else
    Embed_Param.mtry = (size_t) Embed_Param.P * Param.embed_mtry;

  Embed_Param.nmin = Param.embed_nmin;
  Embed_Param.nsplit = Param.embed_nsplit;
  Embed_Param.replacement = Param.embed_replacement;
  Embed_Param.resample_prob = Param.embed_resample_prob;
  Embed_Param.useobsweight = Param.useobsweight;
  Embed_Param.usevarprob = Param.usevarprob;
  Embed_Param.importance = 2; // use distributed variable importance for stability

  Embed_Param.ncores = 1;
  Embed_Param.verbose = 0;
  Embed_Param.seed = rngl.rand_sizet(0, INT_MAX);

  size_t P = Embed_Param.P;
  size_t ntrees = Embed_Param.ntrees;

  imat ObsTrack;

  // initiate uni forest objects
  arma::field<arma::ivec> SplitVar(ntrees);
  arma::field<arma::vec> SplitValue(ntrees);
  arma::field<arma::uvec> LeftNode(ntrees);
  arma::field<arma::uvec> RightNode(ntrees);
  arma::field<arma::vec> NodeWeight(ntrees);
  arma::field<arma::field<arma::vec>> NodeHaz(ntrees);

  // Initiate forest object
  Surv_Uni_Forest_Class SURV_FOREST(SplitVar,
                                     SplitValue,
                                     LeftNode,
                                     RightNode,
                                     NodeWeight,
                                     NodeHaz);

  // Initiate prediction objects
  mat Prediction;

  // VarImp
  vec VarImp(P, fill::zeros);

  // Run model fitting
  vec VarVI_dummy; // not needed for embedded forest
  uvec oob_count_dummy; // not needed for embedded forest

  Surv_Uni_Forest_Build(SURV_DATA,
                        SURV_FOREST,
                        (const PARAM_GLOBAL&) Embed_Param,
                        obs_id,
                        var_id,
                        ObsTrack,
                        true, // do prediction for VI
                        Prediction,
                        oob_count_dummy,
                        VarImp,
                        VarVI_dummy);

  return VarImp;
}
