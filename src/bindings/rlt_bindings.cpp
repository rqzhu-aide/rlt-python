// ************************************************************
// rlt-python: pybind11 bindings replacing the R r-interface layer
// Typed entry points for regression / classification / survival
// forests (fit + predict). numpy arrays in, dict of numpy out.
// Forests are exchanged as plain lists of numpy arrays so that
// fitted models remain picklable (joblib) without C++ support.
// ************************************************************
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "RLT.h"

#include <vector>

namespace py = pybind11;
using namespace arma;

// ------------------------------------------------------------
// c-index (ported from r-interface/cindex.cpp; R-facing versions
// lived in the excluded r-interface layer)
// ------------------------------------------------------------
template <typename Y_INT>
static double cindex_impl(const Y_INT& Y, const arma::uvec& Censor,
                          const arma::vec& pred) {
  size_t P = 0;
  double C = 0;
  for (size_t i = 0; i < Y.n_elem; i++) {
    for (size_t j = 0; j < i; j++) {
      if ((Y(i) > Y(j) && Censor(j) == 0) || (Y(i) < Y(j) && Censor(i) == 0))
        continue;
      if (Y(i) == Y(j) && Censor(i) == 0 && Censor(j) == 0) continue;
      P++;
      if (Y(i) > Y(j)) {
        if (pred(i) < pred(j)) C++;
        if (pred(i) == pred(j)) C += 0.5;
      } else if (Y(i) < Y(j)) {
        if (pred(j) < pred(i)) C++;
        if (pred(i) == pred(j)) C += 0.5;
      } else {
        if (Censor(i) == 1 && Censor(j) == 1) {
          if (pred(i) == pred(j)) C++;
          else C += 0.5;
        } else if ((Censor(i) == 1 && pred(i) > pred(j)) ||
                   (Censor(j) == 1 && pred(j) > pred(i))) {
          C++;
        } else if (pred(i) == pred(j)) {
          C += 0.5;
        }
      }
    }
  }
  return C / (double)P;
}

// ============================================================
// numpy <-> armadillo conversion helpers
// ============================================================

using dmat_t = py::array_t<double, py::array::c_style | py::array::forcecast>;
using imat_t = py::array_t<int64_t, py::array::c_style | py::array::forcecast>;

// (n, p) C-order numpy -> column-major arma::mat (explicit copy).
// NOTE: armadillo's auxiliary-memory constructor always treats the buffer
// as column-major; building a view of C-order memory and calling .t()
// does NOT flip memory layout. Always copy explicitly.
static inline arma::mat np2mat(const dmat_t& a) {
  py::buffer_info bi = a.request();
  if (bi.ndim != 2) throw std::invalid_argument("expected a 2-D array");
  const size_t n = (size_t)bi.shape[0], p = (size_t)bi.shape[1];
  arma::mat X(n, p);
  const double* src = static_cast<const double*>(bi.ptr);
  for (size_t j = 0; j < p; ++j)
    for (size_t i = 0; i < n; ++i) X(i, j) = src[i * p + j];
  return X;
}

// 1-D double numpy -> arma::vec (copy)
static inline arma::vec np2vec(const dmat_t& a) {
  py::buffer_info bi = a.request();
  if (bi.ndim != 1) throw std::invalid_argument("expected a 1-D array");
  const size_t n = (size_t)bi.shape[0];
  arma::vec v(n);
  const double* src = static_cast<const double*>(bi.ptr);
  for (size_t i = 0; i < n; ++i) v(i) = src[i];
  return v;
}

// 1-D int numpy -> arma::uvec (copy, saturating cast)
static inline arma::uvec np2uvec(const imat_t& a) {
  py::buffer_info bi = a.request();
  if (bi.ndim != 1) throw std::invalid_argument("expected a 1-D array");
  const size_t n = (size_t)bi.shape[0];
  arma::uvec v(n);
  const int64_t* src = static_cast<const int64_t*>(bi.ptr);
  for (size_t i = 0; i < n; ++i) v(i) = (uword)src[i];
  return v;
}

// 2-D int numpy -> arma::imat (copy); empty (0,0) allowed
static inline arma::imat np2imat(const imat_t& a) {
  py::buffer_info bi = a.request();
  if (bi.ndim != 2) throw std::invalid_argument("expected a 2-D array");
  const size_t n = (size_t)bi.shape[0], p = (size_t)bi.shape[1];
  arma::imat M(n, p);
  const int64_t* src = static_cast<const int64_t*>(bi.ptr);
  for (size_t j = 0; j < p; ++j)
    for (size_t i = 0; i < n; ++i) M(i, j) = (sword)src[i * p + j];
  return M;
}

static inline py::array_t<double> vec2np(const arma::vec& v) {
  py::array_t<double> out(v.n_elem);
  auto w = out.mutable_unchecked<1>();
  for (size_t i = 0; i < v.n_elem; ++i) w(i) = v(i);
  return out;
}

static inline py::array_t<double> rowvec2np(const arma::rowvec& v) {
  py::array_t<double> out(v.n_elem);
  auto w = out.mutable_unchecked<1>();
  for (size_t i = 0; i < v.n_elem; ++i) w(i) = v(i);
  return out;
}

// column-major arma::mat -> (n, k) C-order numpy (copy)
static inline py::array_t<double> mat2np(const arma::mat& m) {
  py::array_t<double> out({(py::ssize_t)m.n_rows, (py::ssize_t)m.n_cols});
  auto w = out.mutable_unchecked<2>();
  for (size_t i = 0; i < m.n_rows; ++i)
    for (size_t j = 0; j < m.n_cols; ++j) w(i, j) = m(i, j);
  return out;
}

static inline py::array_t<int64_t> ivec2np(const arma::ivec& v) {
  py::array_t<int64_t> out(v.n_elem);
  auto w = out.mutable_unchecked<1>();
  for (size_t i = 0; i < v.n_elem; ++i) w(i) = (int64_t)v(i);
  return out;
}

static inline py::array_t<int64_t> uvec2np(const arma::uvec& v) {
  py::array_t<int64_t> out(v.n_elem);
  auto w = out.mutable_unchecked<1>();
  for (size_t i = 0; i < v.n_elem; ++i) w(i) = (int64_t)v(i);
  return out;
}

static inline py::array_t<int64_t> imat2np(const arma::imat& m) {
  py::array_t<int64_t> out({(py::ssize_t)m.n_rows, (py::ssize_t)m.n_cols});
  auto w = out.mutable_unchecked<2>();
  for (size_t i = 0; i < m.n_rows; ++i)
    for (size_t j = 0; j < m.n_cols; ++j) w(i, j) = (int64_t)m(i, j);
  return out;
}

// arma cube (a, b, c) -> numpy (c0, a, b) where slice index is remapped
// caller-specific; here: direct (a, b, c) C-order copy
static inline py::array_t<double> cube2np(const arma::cube& c) {
  py::array_t<double> out({(py::ssize_t)c.n_rows, (py::ssize_t)c.n_cols,
                           (py::ssize_t)c.n_slices});
  auto w = out.mutable_unchecked<3>();
  for (size_t i = 0; i < c.n_rows; ++i)
    for (size_t j = 0; j < c.n_cols; ++j)
      for (size_t k = 0; k < c.n_slices; ++k) w(i, j, k) = c(i, j, k);
  return out;
}

// ---- python list <-> arma field conversions ----

static inline arma::field<arma::ivec> list2field_ivec(const py::list& lst) {
  arma::field<arma::ivec> f((uword)lst.size());
  for (uword t = 0; t < f.n_elem; ++t) {
    imat_t a = py::cast<imat_t>(lst[t]);
    py::buffer_info bi = a.request();
    if (bi.ndim != 1) throw std::invalid_argument("SplitVar elements must be 1-D");
    arma::ivec v((uword)bi.shape[0]);
    const int64_t* src = static_cast<const int64_t*>(bi.ptr);
    for (uword i = 0; i < v.n_elem; ++i) v(i) = (sword)src[i];
    f(t) = v;
  }
  return f;
}

static inline arma::field<arma::vec> list2field_vec(const py::list& lst) {
  arma::field<arma::vec> f((uword)lst.size());
  for (uword t = 0; t < f.n_elem; ++t) {
    dmat_t a = py::cast<dmat_t>(lst[t]);
    f(t) = np2vec(a);
  }
  return f;
}

static inline arma::field<arma::uvec> list2field_uvec(const py::list& lst) {
  arma::field<arma::uvec> f((uword)lst.size());
  for (uword t = 0; t < f.n_elem; ++t) f(t) = np2uvec(py::cast<imat_t>(lst[t]));
  return f;
}

static inline arma::field<arma::mat> list2field_mat(const py::list& lst) {
  arma::field<arma::mat> f((uword)lst.size());
  for (uword t = 0; t < f.n_elem; ++t) f(t) = np2mat(py::cast<dmat_t>(lst[t]));
  return f;
}

// field<field<vec>>: outer list = trees, inner list = nodes
static inline arma::field<arma::field<arma::vec>> list2field_field_vec(
    const py::list& lst) {
  arma::field<arma::field<arma::vec>> f((uword)lst.size());
  for (uword t = 0; t < f.n_elem; ++t)
    f(t) = list2field_vec(py::cast<py::list>(lst[t]));
  return f;
}

static inline py::list field2list_ivec(const arma::field<arma::ivec>& f) {
  py::list lst((py::ssize_t)f.n_elem);
  for (uword t = 0; t < f.n_elem; ++t) lst[t] = ivec2np(f(t));
  return lst;
}

static inline py::list field2list_vec(const arma::field<arma::vec>& f) {
  py::list lst((py::ssize_t)f.n_elem);
  for (uword t = 0; t < f.n_elem; ++t) lst[t] = vec2np(f(t));
  return lst;
}

static inline py::list field2list_uvec(const arma::field<arma::uvec>& f) {
  py::list lst((py::ssize_t)f.n_elem);
  for (uword t = 0; t < f.n_elem; ++t) lst[t] = uvec2np(f(t));
  return lst;
}

static inline py::list field2list_mat(const arma::field<arma::mat>& f) {
  py::list lst((py::ssize_t)f.n_elem);
  for (uword t = 0; t < f.n_elem; ++t) lst[t] = mat2np(f(t));
  return lst;
}

static inline py::list field2list_field_vec(
    const arma::field<arma::field<arma::vec>>& f) {
  py::list lst((py::ssize_t)f.n_elem);
  for (uword t = 0; t < f.n_elem; ++t) lst[t] = field2list_vec(f(t));
  return lst;
}

static inline arma::field<arma::imat> list2field_imat(const py::list& lst) {
  arma::field<arma::imat> f((uword)lst.size());
  for (uword t = 0; t < f.n_elem; ++t) f(t) = np2imat(py::cast<imat_t>(lst[t]));
  return f;
}

static inline py::list field2list_imat(const arma::field<arma::imat>& f) {
  py::list lst((py::ssize_t)f.n_elem);
  for (uword t = 0; t < f.n_elem; ++t) lst[t] = imat2np(f(t));
  return lst;
}

// ============================================================
// Linear-combination (reinforcement) forests
// ============================================================

// ---- regression Comb ----

static py::dict RegUniCombForestFit(const dmat_t& x, const dmat_t& y,
                                    const imat_t& ncat,
                                    const dmat_t& obsweight,
                                    const dmat_t& varprob,
                                    const imat_t& obstrack,
                                    const rlt::CoreParams& p) {
  arma::mat X = np2mat(x);
  arma::vec Y = np2vec(y);
  arma::uvec Ncat = np2uvec(ncat);
  arma::vec obsw = np2vec(obsweight);
  arma::vec varp = np2vec(varprob);
  arma::imat ObsTrack = np2imat(obstrack);

  PARAM_GLOBAL Param;
  Param.PARAM_READ(p);
  if (Param.verbose) Param.print();

  RLT_REG_DATA REG_DATA(X, Y, Ncat, obsw, varp);

  const size_t N = REG_DATA.X.n_rows;
  const size_t P = REG_DATA.X.n_cols;
  const size_t ntrees = Param.ntrees;
  const int obs_track = Param.obs_track;
  const int importance = Param.importance;

  arma::field<arma::imat> SplitVar(ntrees);
  arma::field<arma::mat> SplitLoad(ntrees);
  arma::field<arma::vec> SplitValue(ntrees);
  arma::field<arma::uvec> LeftNode(ntrees);
  arma::field<arma::uvec> RightNode(ntrees);
  arma::field<arma::vec> NodeWeight(ntrees);
  arma::field<arma::vec> NodeAve(ntrees);

  Reg_Uni_Comb_Forest_Class REG_FOREST(SplitVar, SplitLoad, SplitValue,
                                       LeftNode, RightNode, NodeWeight,
                                       NodeAve);

  uvec obs_id = linspace<uvec>(0, N - 1, N);
  uvec var_id = linspace<uvec>(0, P - 1, P);

  vec Prediction;
  uvec oob_count;
  vec VarImp;
  if (importance) VarImp.zeros(P);
  vec VarVI;
  if (importance && Param.var_mode) VarVI.zeros(P);

  bool do_prediction = Param.replacement or (Param.resample_prob < 1);

  {
    py::gil_scoped_release release;
    Reg_Uni_Comb_Forest_Build((const RLT_REG_DATA&)REG_DATA, REG_FOREST,
                              (const PARAM_GLOBAL&)Param, (const uvec&)obs_id,
                              (const uvec&)var_id, ObsTrack, do_prediction,
                              Prediction, oob_count, VarImp, VarVI);
  }

  py::dict forest;
  forest["SplitVar"] = field2list_imat(SplitVar);
  forest["SplitLoad"] = field2list_mat(SplitLoad);
  forest["SplitValue"] = field2list_vec(SplitValue);
  forest["LeftNode"] = field2list_uvec(LeftNode);
  forest["RightNode"] = field2list_uvec(RightNode);
  forest["NodeWeight"] = field2list_vec(NodeWeight);
  forest["NodeAve"] = field2list_vec(NodeAve);

  py::dict out;
  out["FittedForest"] = forest;
  if (obs_track) out["ObsTrack"] = imat2np(ObsTrack);
  if (importance) out["VarImp"] = vec2np(VarImp);
  if (importance && Param.var_mode == 1) out["VarVI"] = vec2np(VarVI);
  if (Prediction.n_elem > 0) {
    out["Prediction"] = vec2np(Prediction);
    uvec valid = find(oob_count > 0);
    out["Error"] = py::float_(
        (double)mean(square(Prediction(valid) - Y(valid))));
  }
  return out;
}

static py::dict RegUniCombForestPred(const py::list& splitvar,
                                     const py::list& splitload,
                                     const py::list& splitvalue,
                                     const py::list& leftnode,
                                     const py::list& rightnode,
                                     const py::list& nodeweight,
                                     const py::list& nodeave,
                                     const dmat_t& x, const imat_t& ncat,
                                     const imat_t& obstrack, int var_mode,
                                     bool keep_all, size_t usecores,
                                     size_t verbose) {
  usecores = checkCores(usecores, verbose);

  arma::field<arma::imat> SplitVar = list2field_imat(splitvar);
  arma::field<arma::mat> SplitLoad = list2field_mat(splitload);
  arma::field<arma::vec> SplitValue = list2field_vec(splitvalue);
  arma::field<arma::uvec> LeftNode = list2field_uvec(leftnode);
  arma::field<arma::uvec> RightNode = list2field_uvec(rightnode);
  arma::field<arma::vec> NodeWeight = list2field_vec(nodeweight);
  arma::field<arma::vec> NodeAve = list2field_vec(nodeave);
  arma::mat X = np2mat(x);
  arma::uvec Ncat = np2uvec(ncat);
  arma::imat ObsTrack = np2imat(obstrack);

  Reg_Uni_Comb_Forest_Class REG_FOREST(SplitVar, SplitLoad, SplitValue,
                                       LeftNode, RightNode, NodeWeight,
                                       NodeAve);

  const size_t N = X.n_rows;
  const size_t ntrees = REG_FOREST.SplitVarList.size();
  mat PredAll(N, ntrees, fill::zeros);

  {
    py::gil_scoped_release release;
#pragma omp parallel num_threads(usecores)
    {
#pragma omp for schedule(static)
      for (size_t nt = 0; nt < ntrees; nt++) {
        uvec proxy_id = linspace<uvec>(0, N - 1, N);
        uvec real_id = linspace<uvec>(0, N - 1, N);
        uvec TermNode(N, fill::zeros);

        Reg_Uni_Comb_Tree_Class OneTree(REG_FOREST.SplitVarList(nt),
                                        REG_FOREST.SplitLoadList(nt),
                                        REG_FOREST.SplitValueList(nt),
                                        REG_FOREST.LeftNodeList(nt),
                                        REG_FOREST.RightNodeList(nt),
                                        REG_FOREST.NodeWeightList(nt),
                                        REG_FOREST.NodeAveList(nt));

        Find_Terminal_Node_Comb(0, OneTree, X, Ncat, proxy_id, real_id,
                                TermNode);

        PredAll.unsafe_col(nt).rows(real_id) = OneTree.NodeAve(TermNode);
      }
    }
  }

  py::dict out;
  out["Prediction"] = vec2np(mean(PredAll, 1));

  if (var_mode == 1)
    out["Variance"] = vec2np(compute_matched_variance(PredAll));
  else if (var_mode == 2)
    out["Variance"] = vec2np(compute_ij_variance(PredAll, ObsTrack));
  else if (var_mode == 3)
    out["Variance"] = vec2np(compute_jack_variance(PredAll, ObsTrack));

  if (keep_all) out["PredictionAll"] = mat2np(PredAll);
  return out;
}

// ---- classification Comb ----

static py::dict ClaUniCombForestFit(const dmat_t& x, const imat_t& y,
                                    const imat_t& ncat, size_t nclass,
                                    const dmat_t& obsweight,
                                    const dmat_t& varprob,
                                    const imat_t& obstrack,
                                    const rlt::CoreParams& p) {
  arma::mat X = np2mat(x);
  arma::uvec Y = np2uvec(y);
  arma::uvec Ncat = np2uvec(ncat);
  arma::vec obsw = np2vec(obsweight);
  arma::vec varp = np2vec(varprob);
  arma::imat ObsTrack = np2imat(obstrack);

  PARAM_GLOBAL Param;
  Param.PARAM_READ(p);
  if (Param.verbose) Param.print();

  RLT_CLA_DATA CLA_DATA(X, Y, Ncat, nclass, obsw, varp);

  const size_t N = CLA_DATA.X.n_rows;
  const size_t P = CLA_DATA.X.n_cols;
  const size_t ntrees = Param.ntrees;
  const int obs_track = Param.obs_track;
  const int importance = Param.importance;

  arma::field<arma::imat> SplitVar(ntrees);
  arma::field<arma::mat> SplitLoad(ntrees);
  arma::field<arma::vec> SplitValue(ntrees);
  arma::field<arma::uvec> LeftNode(ntrees);
  arma::field<arma::uvec> RightNode(ntrees);
  arma::field<arma::vec> NodeWeight(ntrees);
  arma::field<arma::mat> NodeProb(ntrees);

  Cla_Uni_Comb_Forest_Class CLA_FOREST(SplitVar, SplitLoad, SplitValue,
                                       LeftNode, RightNode, NodeWeight,
                                       NodeProb);

  uvec obs_id = linspace<uvec>(0, N - 1, N);
  uvec var_id = linspace<uvec>(0, P - 1, P);

  mat Prediction;
  uvec oob_count;
  vec VarImp;
  if (importance) VarImp.zeros(P);
  vec VarVI;
  if (importance && Param.var_mode) VarVI.zeros(P);

  bool do_prediction = Param.replacement or (Param.resample_prob < 1);

  {
    py::gil_scoped_release release;
    Cla_Uni_Comb_Forest_Build((const RLT_CLA_DATA&)CLA_DATA, CLA_FOREST,
                              (const PARAM_GLOBAL&)Param, (const uvec&)obs_id,
                              (const uvec&)var_id, ObsTrack, do_prediction,
                              Prediction, oob_count, VarImp, VarVI);
  }

  py::dict forest;
  forest["SplitVar"] = field2list_imat(SplitVar);
  forest["SplitLoad"] = field2list_mat(SplitLoad);
  forest["SplitValue"] = field2list_vec(SplitValue);
  forest["LeftNode"] = field2list_uvec(LeftNode);
  forest["RightNode"] = field2list_uvec(RightNode);
  forest["NodeWeight"] = field2list_vec(NodeWeight);
  forest["NodeProb"] = field2list_mat(NodeProb);

  py::dict out;
  out["FittedForest"] = forest;
  if (obs_track) out["ObsTrack"] = imat2np(ObsTrack);
  if (importance) out["VarImp"] = vec2np(VarImp);
  if (importance && Param.var_mode == 1) out["VarVI"] = vec2np(VarVI);
  if (Prediction.n_elem > 0) {
    uvec PredClass = index_max(Prediction, 1);
    out["Prediction"] = uvec2np(PredClass);
    out["Prob"] = mat2np(Prediction);
    uvec valid = find(oob_count > 0);
    out["Error"] = py::float_(
        (double)sum(Y(valid) != PredClass(valid)) / (double)valid.n_elem);
  }
  return out;
}

static py::dict ClaUniCombForestPred(const py::list& splitvar,
                                     const py::list& splitload,
                                     const py::list& splitvalue,
                                     const py::list& leftnode,
                                     const py::list& rightnode,
                                     const py::list& nodeweight,
                                     const py::list& nodeprob,
                                     const dmat_t& x, const imat_t& ncat,
                                     const imat_t& obstrack, int var_mode,
                                     bool keep_all, size_t usecores,
                                     size_t verbose) {
  usecores = checkCores(usecores, verbose);

  arma::field<arma::imat> SplitVar = list2field_imat(splitvar);
  arma::field<arma::mat> SplitLoad = list2field_mat(splitload);
  arma::field<arma::vec> SplitValue = list2field_vec(splitvalue);
  arma::field<arma::uvec> LeftNode = list2field_uvec(leftnode);
  arma::field<arma::uvec> RightNode = list2field_uvec(rightnode);
  arma::field<arma::vec> NodeWeight = list2field_vec(nodeweight);
  arma::field<arma::mat> NodeProb = list2field_mat(nodeprob);
  arma::mat X = np2mat(x);
  arma::uvec Ncat = np2uvec(ncat);
  arma::imat ObsTrack = np2imat(obstrack);

  Cla_Uni_Comb_Forest_Class CLA_FOREST(SplitVar, SplitLoad, SplitValue,
                                       LeftNode, RightNode, NodeWeight,
                                       NodeProb);

  const size_t N = X.n_rows;
  const size_t ntrees = CLA_FOREST.SplitVarList.size();
  const size_t nclass = CLA_FOREST.NodeProbList(0).n_cols;

  cube PredAll(ntrees, nclass, N, fill::zeros);
  mat Prob(N, nclass, fill::zeros);
  uvec Pred(N, fill::zeros);

  {
    py::gil_scoped_release release;
#pragma omp parallel num_threads(usecores)
    {
#pragma omp for schedule(static)
      for (size_t nt = 0; nt < ntrees; nt++) {
        uvec proxy_id = linspace<uvec>(0, N - 1, N);
        uvec real_id = linspace<uvec>(0, N - 1, N);
        uvec TermNode(N, fill::zeros);

        Cla_Uni_Comb_Tree_Class OneTree(CLA_FOREST.SplitVarList(nt),
                                        CLA_FOREST.SplitLoadList(nt),
                                        CLA_FOREST.SplitValueList(nt),
                                        CLA_FOREST.LeftNodeList(nt),
                                        CLA_FOREST.RightNodeList(nt),
                                        CLA_FOREST.NodeWeightList(nt),
                                        CLA_FOREST.NodeProbList(nt));

        Find_Terminal_Node_Comb(0, OneTree, X, Ncat, proxy_id, real_id,
                                TermNode);

        for (size_t i = 0; i < N; i++)
          PredAll.slice(i).row(nt) =
              CLA_FOREST.NodeProbList(nt).row(TermNode(i));
      }

#pragma omp barrier
#pragma omp for schedule(static)
      for (size_t i = 0; i < N; i++) {
        Prob.row(i) = mean(PredAll.slice(i), 0);
        Pred(i) = index_max(Prob.row(i));
      }
    }
  }

  py::dict out;
  out["Prediction"] = uvec2np(Pred);
  out["Prob"] = mat2np(Prob);
  if (keep_all) {
    py::array_t<double> all({(py::ssize_t)N, (py::ssize_t)ntrees,
                             (py::ssize_t)nclass});
    auto w = all.mutable_unchecked<3>();
    for (size_t i = 0; i < N; ++i)
      for (size_t nt = 0; nt < ntrees; ++nt)
        for (size_t k = 0; k < nclass; ++k) w(i, nt, k) = PredAll(nt, k, i);
    out["PredictionAll"] = all;
  }
  return out;
}

// ---- survival Comb ----

static py::dict SurvUniCombForestFit(const dmat_t& x, const imat_t& y,
                                     const imat_t& censor, const imat_t& ncat,
                                     const dmat_t& obsweight,
                                     const dmat_t& varprob,
                                     const imat_t& obstrack,
                                     const rlt::CoreParams& p) {
  arma::mat X = np2mat(x);
  arma::uvec Y = np2uvec(y);
  arma::uvec Censor = np2uvec(censor);
  arma::uvec Ncat = np2uvec(ncat);
  arma::vec obsw = np2vec(obsweight);
  arma::vec varp = np2vec(varprob);
  arma::imat ObsTrack = np2imat(obstrack);

  PARAM_GLOBAL Param;
  Param.PARAM_READ(p);
  if (Param.verbose) Param.print();

  uvec failY = Y(find(Censor == 1));
  if (failY.is_empty())
    throw std::invalid_argument("no observed failures (censor all zero)");
  size_t NFail = (size_t)max(failY);

  RLT_SURV_DATA SURV_DATA(X, Y, Censor, Ncat, NFail, obsw, varp);

  const size_t N = SURV_DATA.X.n_rows;
  const size_t P = SURV_DATA.X.n_cols;
  const size_t ntrees = Param.ntrees;
  const int obs_track = Param.obs_track;
  const int importance = Param.importance;

  arma::field<arma::imat> SplitVar(ntrees);
  arma::field<arma::mat> SplitLoad(ntrees);
  arma::field<arma::vec> SplitValue(ntrees);
  arma::field<arma::uvec> LeftNode(ntrees);
  arma::field<arma::uvec> RightNode(ntrees);
  arma::field<arma::vec> NodeWeight(ntrees);
  arma::field<arma::field<arma::vec>> NodeHaz(ntrees);

  Surv_Uni_Comb_Forest_Class SURV_FOREST(SplitVar, SplitLoad, SplitValue,
                                         LeftNode, RightNode, NodeWeight,
                                         NodeHaz);

  uvec obs_id = linspace<uvec>(0, N - 1, N);
  uvec var_id = linspace<uvec>(0, P - 1, P);

  mat Prediction;
  uvec oob_count;
  vec VarImp;
  if (importance) VarImp.zeros(P);
  vec VarVI;
  if (importance && Param.var_mode) VarVI.zeros(P);

  bool do_prediction = Param.replacement or (Param.resample_prob < 1);

  {
    py::gil_scoped_release release;
    Surv_Uni_Comb_Forest_Build((const RLT_SURV_DATA&)SURV_DATA, SURV_FOREST,
                               (const PARAM_GLOBAL&)Param, (const uvec&)obs_id,
                               (const uvec&)var_id, ObsTrack, do_prediction,
                               Prediction, oob_count, VarImp, VarVI);
  }

  py::dict forest;
  forest["SplitVar"] = field2list_imat(SplitVar);
  forest["SplitLoad"] = field2list_mat(SplitLoad);
  forest["SplitValue"] = field2list_vec(SplitValue);
  forest["LeftNode"] = field2list_uvec(LeftNode);
  forest["RightNode"] = field2list_uvec(RightNode);
  forest["NodeWeight"] = field2list_vec(NodeWeight);
  forest["NodeHaz"] = field2list_field_vec(NodeHaz);

  py::dict out;
  out["FittedForest"] = forest;
  out["NFail"] = py::int_((int)NFail);
  if (obs_track) out["ObsTrack"] = imat2np(ObsTrack);
  if (importance) out["VarImp"] = vec2np(VarImp);
  if (importance && Param.var_mode == 1) out["VarVI"] = vec2np(VarVI);
  if (Prediction.n_elem > 0) {
    out["Prediction"] = mat2np(Prediction);
    uvec valid = find(oob_count > 0);
    vec oobcch(Prediction.n_rows, fill::zeros);
    for (size_t k = 0; k < valid.n_elem; k++)
      oobcch(k) = accu(cumsum(Prediction.row(valid(k))));
    uvec subY = Y(valid);
    uvec subCensor = Censor(valid);
    out["Error"] = py::float_(1.0 - cindex_impl<arma::uvec>(subY, subCensor,
                                                            oobcch));
  }
  return out;
}

static py::dict SurvUniCombForestPred(const py::list& splitvar,
                                      const py::list& splitload,
                                      const py::list& splitvalue,
                                      const py::list& leftnode,
                                      const py::list& rightnode,
                                      const py::list& nodeweight,
                                      const py::list& nodehaz,
                                      const dmat_t& x, const imat_t& ncat,
                                      size_t NFail,
                                      const imat_t& mapping_indices,
                                      const imat_t& obstrack, int var_mode,
                                      bool keep_all, size_t usecores,
                                      size_t verbose) {
  usecores = checkCores(usecores, verbose);

  arma::field<arma::imat> SplitVar = list2field_imat(splitvar);
  arma::field<arma::mat> SplitLoad = list2field_mat(splitload);
  arma::field<arma::vec> SplitValue = list2field_vec(splitvalue);
  arma::field<arma::uvec> LeftNode = list2field_uvec(leftnode);
  arma::field<arma::uvec> RightNode = list2field_uvec(rightnode);
  arma::field<arma::vec> NodeWeight = list2field_vec(nodeweight);
  arma::field<arma::field<arma::vec>> NodeHaz =
      list2field_field_vec(nodehaz);
  arma::mat X = np2mat(x);
  arma::uvec Ncat = np2uvec(ncat);
  arma::uvec mapping = np2uvec(mapping_indices);
  arma::imat ObsTrack = np2imat(obstrack);

  Surv_Uni_Comb_Forest_Class SURV_FOREST(SplitVar, SplitLoad, SplitValue,
                                         LeftNode, RightNode, NodeWeight,
                                         NodeHaz);

  const size_t N = X.n_rows;
  const size_t ntrees = SURV_FOREST.SplitVarList.size();
  bool VarEst = (var_mode > 0);
  const size_t effective_grid_size = mapping.n_elem;

  umat AllTermNode(N, ntrees, fill::zeros);

  mat Hazard(N, effective_grid_size);
  mat CHazard(N, effective_grid_size);
  mat Surv(N, effective_grid_size);

  cube Cov;
  if (VarEst) Cov.zeros(effective_grid_size, effective_grid_size, N);

  cube AllHazard;
  if (keep_all) AllHazard.zeros(ntrees, effective_grid_size, N);

  {
    py::gil_scoped_release release;
#pragma omp parallel num_threads(usecores)
    {
#pragma omp for schedule(static)
      for (size_t nt = 0; nt < ntrees; nt++) {
        uvec proxy_id = linspace<uvec>(0, N - 1, N);
        uvec real_id = linspace<uvec>(0, N - 1, N);
        uvec TermNode(N, fill::zeros);
        Comb_Tree_Class OneTree(SURV_FOREST.SplitVarList(nt),
                                SURV_FOREST.SplitLoadList(nt),
                                SURV_FOREST.SplitValueList(nt),
                                SURV_FOREST.LeftNodeList(nt),
                                SURV_FOREST.RightNodeList(nt),
                                SURV_FOREST.NodeWeightList(nt));
        Find_Terminal_Node_Comb(0, OneTree, X, Ncat, proxy_id, real_id,
                                TermNode);
        AllTermNode.col(nt) = TermNode;
      }

#pragma omp barrier

#pragma omp for schedule(static)
      for (size_t i = 0; i < N; i++) {
        mat pred_i_hazard_full(ntrees, NFail);
        for (size_t nt = 0; nt < ntrees; nt++) {
          arma::vec H_full = SURV_FOREST.NodeHazList(nt).at(AllTermNode(i, nt));
          pred_i_hazard_full.row(nt) = H_full.subvec(1, NFail).t();
        }

        mat pred_i_survival_full = cumprod(1 - pred_i_hazard_full, 1);

        mat pred_i_hazard_reduced = pred_i_hazard_full.cols(mapping);

        Hazard.row(i) = mean(pred_i_hazard_reduced, 0);
        CHazard.row(i) = cumsum(Hazard.row(i));
        Surv.row(i) = exp(-CHazard.row(i));

        if (VarEst) {
          mat pred_i_for_var = cumsum(pred_i_hazard_reduced, 1);

          if (var_mode == 1) {
            size_t B = ntrees / 2;
            mat Diff =
                pred_i_for_var.rows(0, B - 1) - pred_i_for_var.rows(B, ntrees - 1);
            mat Vh = Diff.t() * Diff / ntrees;
            mat Vs = cov(pred_i_for_var, 1);
            mat diffmat = Vh - Vs;
            vec eigval;
            mat eigvec;
            eig_sym(eigval, eigvec, diffmat);
            eigval.elem(find(eigval < 1e-6)).fill(1e-6);
            Cov.slice(i) = eigvec * diagmat(eigval) * eigvec.t();
            Cov.slice(i) = (Cov.slice(i) + Cov.slice(i).t()) / 2;
          } else if (var_mode == 2) {
            Cov.slice(i) = compute_surv_ij_variance(pred_i_for_var, ObsTrack);
          } else if (var_mode == 3) {
            Cov.slice(i) = compute_surv_jack_variance(pred_i_for_var, ObsTrack);
          }
        }

        if (keep_all)
          AllHazard.slice(i).cols(0, effective_grid_size - 1) =
              pred_i_hazard_reduced;
      }
    }
  }

  py::dict out;
  out["Hazard"] = mat2np(Hazard);
  out["CHF"] = mat2np(CHazard);
  out["Survival"] = mat2np(Surv);
  if (VarEst) {
    py::array_t<double> cov({(py::ssize_t)N,
                             (py::ssize_t)effective_grid_size,
                             (py::ssize_t)effective_grid_size});
    auto w = cov.mutable_unchecked<3>();
    for (size_t i = 0; i < N; ++i)
      for (size_t a = 0; a < effective_grid_size; ++a)
        for (size_t b = 0; b < effective_grid_size; ++b)
          w(i, a, b) = Cov(a, b, i);
    out["Cov"] = cov;
  }
  if (keep_all) {
    py::array_t<double> all({(py::ssize_t)N, (py::ssize_t)ntrees,
                             (py::ssize_t)effective_grid_size});
    auto w = all.mutable_unchecked<3>();
    for (size_t i = 0; i < N; ++i)
      for (size_t nt = 0; nt < ntrees; ++nt)
        for (size_t t = 0; t < effective_grid_size; ++t)
          w(i, nt, t) = AllHazard(nt, t, i);
    out["AllHazard"] = all;
  }
  return out;
}

// ============================================================
// Regression forest
// ============================================================

static py::dict RegUniForestFit(const dmat_t& x, const dmat_t& y,
                                const imat_t& ncat, const dmat_t& obsweight,
                                const dmat_t& varprob, const imat_t& obstrack,
                                const rlt::CoreParams& p) {
  arma::mat X = np2mat(x);
  arma::vec Y = np2vec(y);
  arma::uvec Ncat = np2uvec(ncat);
  arma::vec obsw = np2vec(obsweight);
  arma::vec varp = np2vec(varprob);
  arma::imat ObsTrack = np2imat(obstrack);

  PARAM_GLOBAL Param;
  Param.PARAM_READ(p);
  if (Param.verbose) Param.print();

  RLT_REG_DATA REG_DATA(X, Y, Ncat, obsw, varp);

  const size_t N = REG_DATA.X.n_rows;
  const size_t P = REG_DATA.X.n_cols;
  const size_t ntrees = Param.ntrees;
  const int obs_track = Param.obs_track;
  const int importance = Param.importance;

  arma::field<arma::ivec> SplitVar(ntrees);
  arma::field<arma::vec> SplitValue(ntrees);
  arma::field<arma::uvec> LeftNode(ntrees);
  arma::field<arma::uvec> RightNode(ntrees);
  arma::field<arma::vec> NodeWeight(ntrees);
  arma::field<arma::vec> NodeAve(ntrees);

  Reg_Uni_Forest_Class REG_FOREST(SplitVar, SplitValue, LeftNode, RightNode,
                                  NodeWeight, NodeAve);

  uvec obs_id = linspace<uvec>(0, N - 1, N);
  uvec var_id = linspace<uvec>(0, P - 1, P);

  vec Prediction;
  uvec oob_count;

  vec VarImp;
  if (importance) VarImp.zeros(P);

  vec VarVI;
  if (importance && Param.var_mode) VarVI.zeros(P);

  bool do_prediction = Param.replacement or (Param.resample_prob < 1);

  {
    py::gil_scoped_release release;
    Reg_Uni_Forest_Build((const RLT_REG_DATA&)REG_DATA, REG_FOREST,
                         (const PARAM_GLOBAL&)Param, (const uvec&)obs_id,
                         (const uvec&)var_id, ObsTrack, do_prediction,
                         Prediction, oob_count, VarImp, VarVI);
  }

  py::dict forest;
  forest["SplitVar"] = field2list_ivec(SplitVar);
  forest["SplitValue"] = field2list_vec(SplitValue);
  forest["LeftNode"] = field2list_uvec(LeftNode);
  forest["RightNode"] = field2list_uvec(RightNode);
  forest["NodeWeight"] = field2list_vec(NodeWeight);
  forest["NodeAve"] = field2list_vec(NodeAve);

  py::dict out;
  out["FittedForest"] = forest;
  if (obs_track) out["ObsTrack"] = imat2np(ObsTrack);
  if (importance) out["VarImp"] = vec2np(VarImp);
  if (importance && Param.var_mode == 1) out["VarVI"] = vec2np(VarVI);
  if (Prediction.n_elem > 0) {
    out["Prediction"] = vec2np(Prediction);
    uvec valid = find(oob_count > 0);
    out["Error"] = py::float_(
        (double)mean(square(Prediction(valid) - Y(valid))));
  }
  return out;
}

static py::dict RegUniForestPred(const py::list& splitvar,
                                 const py::list& splitvalue,
                                 const py::list& leftnode,
                                 const py::list& rightnode,
                                 const py::list& nodeweight,
                                 const py::list& nodeave, const dmat_t& x,
                                 const imat_t& ncat, const imat_t& obstrack,
                                 int var_mode, bool keep_all, size_t usecores,
                                 size_t verbose) {
  usecores = checkCores(usecores, verbose);

  arma::field<arma::ivec> SplitVar = list2field_ivec(splitvar);
  arma::field<arma::vec> SplitValue = list2field_vec(splitvalue);
  arma::field<arma::uvec> LeftNode = list2field_uvec(leftnode);
  arma::field<arma::uvec> RightNode = list2field_uvec(rightnode);
  arma::field<arma::vec> NodeWeight = list2field_vec(nodeweight);
  arma::field<arma::vec> NodeAve = list2field_vec(nodeave);
  arma::mat X = np2mat(x);
  arma::uvec Ncat = np2uvec(ncat);
  arma::imat ObsTrack = np2imat(obstrack);

  Reg_Uni_Forest_Class REG_FOREST(SplitVar, SplitValue, LeftNode, RightNode,
                                  NodeWeight, NodeAve);

  const size_t N = X.n_rows;
  const size_t ntrees = REG_FOREST.SplitVarList.size();
  mat PredAll(N, ntrees, fill::zeros);

  {
    py::gil_scoped_release release;
#pragma omp parallel num_threads(usecores)
    {
#pragma omp for schedule(static)
      for (size_t nt = 0; nt < ntrees; nt++) {
        uvec proxy_id = linspace<uvec>(0, N - 1, N);
        uvec real_id = linspace<uvec>(0, N - 1, N);
        uvec TermNode(N, fill::zeros);

        Reg_Uni_Tree_Class OneTree(REG_FOREST.SplitVarList(nt),
                                   REG_FOREST.SplitValueList(nt),
                                   REG_FOREST.LeftNodeList(nt),
                                   REG_FOREST.RightNodeList(nt),
                                   REG_FOREST.NodeWeightList(nt),
                                   REG_FOREST.NodeAveList(nt));

        Find_Terminal_Node(0, OneTree, X, Ncat, proxy_id, real_id, TermNode);

        PredAll.unsafe_col(nt).rows(real_id) = OneTree.NodeAve(TermNode);
      }
    }
  }

  py::dict out;
  out["Prediction"] = vec2np(mean(PredAll, 1));

  if (var_mode == 1)
    out["Variance"] = vec2np(compute_matched_variance(PredAll));
  else if (var_mode == 2)
    out["Variance"] = vec2np(compute_ij_variance(PredAll, ObsTrack));
  else if (var_mode == 3)
    out["Variance"] = vec2np(compute_jack_variance(PredAll, ObsTrack));

  if (keep_all) out["PredictionAll"] = mat2np(PredAll);
  return out;
}

// ============================================================
// Classification forest
// ============================================================

static py::dict ClaUniForestFit(const dmat_t& x, const imat_t& y,
                                const imat_t& ncat, size_t nclass,
                                const dmat_t& obsweight, const dmat_t& varprob,
                                const imat_t& obstrack,
                                const rlt::CoreParams& p) {
  arma::mat X = np2mat(x);
  arma::uvec Y = np2uvec(y);
  arma::uvec Ncat = np2uvec(ncat);
  arma::vec obsw = np2vec(obsweight);
  arma::vec varp = np2vec(varprob);
  arma::imat ObsTrack = np2imat(obstrack);

  PARAM_GLOBAL Param;
  Param.PARAM_READ(p);
  if (Param.verbose) Param.print();

  RLT_CLA_DATA CLA_DATA(X, Y, Ncat, nclass, obsw, varp);

  const size_t N = CLA_DATA.X.n_rows;
  const size_t P = CLA_DATA.X.n_cols;
  const size_t ntrees = Param.ntrees;
  const int obs_track = Param.obs_track;
  const int importance = Param.importance;

  arma::field<arma::ivec> SplitVar(ntrees);
  arma::field<arma::vec> SplitValue(ntrees);
  arma::field<arma::uvec> LeftNode(ntrees);
  arma::field<arma::uvec> RightNode(ntrees);
  arma::field<arma::vec> NodeWeight(ntrees);
  arma::field<arma::mat> NodeProb(ntrees);

  Cla_Uni_Forest_Class CLA_FOREST(SplitVar, SplitValue, LeftNode, RightNode,
                                  NodeWeight, NodeProb);

  uvec obs_id = linspace<uvec>(0, N - 1, N);
  uvec var_id = linspace<uvec>(0, P - 1, P);

  mat Prediction;
  uvec oob_count;

  bool do_prediction = Param.replacement or (Param.resample_prob < 1);

  vec VarImp;
  if (importance) VarImp.zeros(P);
  vec VarVI;
  if (importance && Param.var_mode) VarVI.zeros(P);

  {
    py::gil_scoped_release release;
    Cla_Uni_Forest_Build((const RLT_CLA_DATA&)CLA_DATA, CLA_FOREST,
                         (const PARAM_GLOBAL&)Param, (const uvec&)obs_id,
                         (const uvec&)var_id, ObsTrack, do_prediction,
                         Prediction, oob_count, VarImp, VarVI);
  }

  py::dict forest;
  forest["SplitVar"] = field2list_ivec(SplitVar);
  forest["SplitValue"] = field2list_vec(SplitValue);
  forest["LeftNode"] = field2list_uvec(LeftNode);
  forest["RightNode"] = field2list_uvec(RightNode);
  forest["NodeWeight"] = field2list_vec(NodeWeight);
  forest["NodeProb"] = field2list_mat(NodeProb);

  py::dict out;
  out["FittedForest"] = forest;
  if (obs_track) out["ObsTrack"] = imat2np(ObsTrack);
  if (importance) out["VarImp"] = vec2np(VarImp);
  if (importance && Param.var_mode == 1) out["VarVI"] = vec2np(VarVI);
  if (Prediction.n_elem > 0) {
    uvec PredClass = index_max(Prediction, 1);
    out["Prediction"] = uvec2np(PredClass);
    out["Prob"] = mat2np(Prediction);
    uvec valid = find(oob_count > 0);
    out["Error"] = py::float_(
        (double)sum(PredClass(valid) != Y(valid)) / (double)valid.n_elem);
  }
  return out;
}

static py::dict ClaUniForestPred(const py::list& splitvar,
                                 const py::list& splitvalue,
                                 const py::list& leftnode,
                                 const py::list& rightnode,
                                 const py::list& nodeweight,
                                 const py::list& nodeprob, const dmat_t& x,
                                 const imat_t& ncat, const imat_t& obstrack,
                                 int var_mode, bool keep_all, size_t usecores,
                                 size_t verbose) {
  usecores = checkCores(usecores, verbose);

  arma::field<arma::ivec> SplitVar = list2field_ivec(splitvar);
  arma::field<arma::vec> SplitValue = list2field_vec(splitvalue);
  arma::field<arma::uvec> LeftNode = list2field_uvec(leftnode);
  arma::field<arma::uvec> RightNode = list2field_uvec(rightnode);
  arma::field<arma::vec> NodeWeight = list2field_vec(nodeweight);
  arma::field<arma::mat> NodeProb = list2field_mat(nodeprob);
  arma::mat X = np2mat(x);
  arma::uvec Ncat = np2uvec(ncat);
  arma::imat ObsTrack = np2imat(obstrack);

  Cla_Uni_Forest_Class CLA_FOREST(SplitVar, SplitValue, LeftNode, RightNode,
                                  NodeWeight, NodeProb);

  const size_t N = X.n_rows;
  const size_t ntrees = CLA_FOREST.SplitVarList.size();
  const size_t nclass = CLA_FOREST.NodeProbList(0).n_cols;

  cube PredAll(ntrees, nclass, N, fill::zeros);
  mat Prob(N, nclass, fill::zeros);
  uvec Pred(N, fill::zeros);

  {
    py::gil_scoped_release release;
#pragma omp parallel num_threads(usecores)
    {
#pragma omp for schedule(static)
      for (size_t nt = 0; nt < ntrees; nt++) {
        uvec proxy_id = linspace<uvec>(0, N - 1, N);
        uvec real_id = linspace<uvec>(0, N - 1, N);
        uvec TermNode(N, fill::zeros);

        Tree_Class OneTree(CLA_FOREST.SplitVarList(nt),
                           CLA_FOREST.SplitValueList(nt),
                           CLA_FOREST.LeftNodeList(nt),
                           CLA_FOREST.RightNodeList(nt),
                           CLA_FOREST.NodeWeightList(nt));

        Find_Terminal_Node(0, OneTree, X, Ncat, proxy_id, real_id, TermNode);

        for (size_t i = 0; i < N; i++)
          PredAll.slice(i).row(nt) = CLA_FOREST.NodeProbList(nt).row(TermNode(i));
      }

#pragma omp barrier
#pragma omp for schedule(static)
      for (size_t i = 0; i < N; i++) {
        Prob.row(i) = mean(PredAll.slice(i), 0);
        Pred(i) = index_max(Prob.row(i));
      }
    }
  }

  py::dict out;
  out["Prediction"] = uvec2np(Pred);
  out["Prob"] = mat2np(Prob);
  if (keep_all) {
    // return per-subject layout: (N, ntrees, nclass)
    py::array_t<double> all({(py::ssize_t)N, (py::ssize_t)ntrees,
                             (py::ssize_t)nclass});
    auto w = all.mutable_unchecked<3>();
    for (size_t i = 0; i < N; ++i)
      for (size_t nt = 0; nt < ntrees; ++nt)
        for (size_t k = 0; k < nclass; ++k) w(i, nt, k) = PredAll(nt, k, i);
    out["PredictionAll"] = all;
  }
  return out;
}

// ============================================================
// Survival forest
// ============================================================

static py::dict SurvUniForestFit(const dmat_t& x, const imat_t& y,
                                 const imat_t& censor, const imat_t& ncat,
                                 const dmat_t& obsweight, const dmat_t& varprob,
                                 const imat_t& obstrack,
                                 const rlt::CoreParams& p) {
  arma::mat X = np2mat(x);
  arma::uvec Y = np2uvec(y);
  arma::uvec Censor = np2uvec(censor);
  arma::uvec Ncat = np2uvec(ncat);
  arma::vec obsw = np2vec(obsweight);
  arma::vec varp = np2vec(varprob);
  arma::imat ObsTrack = np2imat(obstrack);

  PARAM_GLOBAL Param;
  Param.PARAM_READ(p);
  if (Param.verbose) Param.print();

  uvec failY = Y(find(Censor == 1));
  if (failY.is_empty())
    throw std::invalid_argument("no observed failures (censor all zero)");
  size_t NFail = (size_t)max(failY);

  RLT_SURV_DATA SURV_DATA(X, Y, Censor, Ncat, NFail, obsw, varp);

  const size_t N = SURV_DATA.X.n_rows;
  const size_t P = SURV_DATA.X.n_cols;
  const size_t ntrees = Param.ntrees;
  const int obs_track = Param.obs_track;
  const int importance = Param.importance;

  arma::field<arma::ivec> SplitVar(ntrees);
  arma::field<arma::vec> SplitValue(ntrees);
  arma::field<arma::uvec> LeftNode(ntrees);
  arma::field<arma::uvec> RightNode(ntrees);
  arma::field<arma::vec> NodeWeight(ntrees);
  arma::field<arma::field<arma::vec>> NodeHaz(ntrees);

  Surv_Uni_Forest_Class SURV_FOREST(SplitVar, SplitValue, LeftNode, RightNode,
                                    NodeWeight, NodeHaz);

  uvec obs_id = linspace<uvec>(0, N - 1, N);
  uvec var_id = linspace<uvec>(0, P - 1, P);

  mat Prediction;
  uvec oob_count;

  vec VarImp;
  if (importance) VarImp.zeros(P);
  vec VarVI;
  if (importance && Param.var_mode) VarVI.zeros(P);

  bool do_prediction = Param.replacement or (Param.resample_prob < 1);

  {
    py::gil_scoped_release release;
    Surv_Uni_Forest_Build((const RLT_SURV_DATA&)SURV_DATA, SURV_FOREST,
                          (const PARAM_GLOBAL&)Param, (const uvec&)obs_id,
                          (const uvec&)var_id, ObsTrack, do_prediction,
                          Prediction, oob_count, VarImp, VarVI);
  }

  py::dict forest;
  forest["SplitVar"] = field2list_ivec(SplitVar);
  forest["SplitValue"] = field2list_vec(SplitValue);
  forest["LeftNode"] = field2list_uvec(LeftNode);
  forest["RightNode"] = field2list_uvec(RightNode);
  forest["NodeWeight"] = field2list_vec(NodeWeight);
  forest["NodeHaz"] = field2list_field_vec(NodeHaz);

  py::dict out;
  out["FittedForest"] = forest;
  if (obs_track) out["ObsTrack"] = imat2np(ObsTrack);
  if (importance) out["VarImp"] = vec2np(VarImp);
  if (importance && Param.var_mode == 1) out["VarVI"] = vec2np(VarVI);

  if (Prediction.n_elem > 0) {
    out["Prediction"] = mat2np(Prediction);

    uvec valid = find(oob_count > 0);
    vec oobcch(valid.n_elem, fill::zeros);
    for (size_t k = 0; k < valid.n_elem; k++)
      oobcch(k) = accu(cumsum(Prediction.row(valid(k))));

    uvec subY = Y(valid);
    uvec subCensor = Censor(valid);
    out["Error"] = py::float_(1.0 - cindex_impl<arma::uvec>(subY, subCensor,
                                                            oobcch));
  }

  out["NFail"] = py::int_((int64_t)NFail);
  return out;
}

static py::dict SurvUniForestPred(const py::list& splitvar,
                                  const py::list& splitvalue,
                                  const py::list& leftnode,
                                  const py::list& rightnode,
                                  const py::list& nodeweight,
                                  const py::list& nodehaz, const dmat_t& x,
                                  const imat_t& ncat, size_t NFail,
                                  const imat_t& mapping_indices,
                                  const imat_t& obstrack, int var_mode,
                                  bool keep_all, size_t usecores,
                                  size_t verbose) {
  usecores = checkCores(usecores, verbose);

  arma::field<arma::ivec> SplitVar = list2field_ivec(splitvar);
  arma::field<arma::vec> SplitValue = list2field_vec(splitvalue);
  arma::field<arma::uvec> LeftNode = list2field_uvec(leftnode);
  arma::field<arma::uvec> RightNode = list2field_uvec(rightnode);
  arma::field<arma::vec> NodeWeight = list2field_vec(nodeweight);
  arma::field<arma::field<arma::vec>> NodeHaz = list2field_field_vec(nodehaz);
  arma::mat X = np2mat(x);
  arma::uvec Ncat = np2uvec(ncat);
  arma::uvec mapping = np2uvec(mapping_indices);
  arma::imat ObsTrack = np2imat(obstrack);

  Surv_Uni_Forest_Class SURV_FOREST(SplitVar, SplitValue, LeftNode, RightNode,
                                    NodeWeight, NodeHaz);

  const size_t N = X.n_rows;
  const size_t ntrees = SURV_FOREST.SplitVarList.size();
  bool VarEst = (var_mode > 0);
  const size_t effective_grid_size = mapping.n_elem;

  umat AllTermNode(N, ntrees, fill::zeros);

  mat Hazard(N, effective_grid_size);
  mat CHazard(N, effective_grid_size);
  mat Surv(N, effective_grid_size);

  cube Cov;
  if (VarEst) Cov.zeros(effective_grid_size, effective_grid_size, N);

  cube AllHazard;
  if (keep_all) AllHazard.zeros(ntrees, effective_grid_size, N);

  {
    py::gil_scoped_release release;
#pragma omp parallel num_threads(usecores)
    {
#pragma omp for schedule(static)
      for (size_t nt = 0; nt < ntrees; nt++) {
        uvec proxy_id = linspace<uvec>(0, N - 1, N);
        uvec real_id = linspace<uvec>(0, N - 1, N);
        uvec TermNode(N, fill::zeros);
        Tree_Class OneTree(SURV_FOREST.SplitVarList(nt),
                           SURV_FOREST.SplitValueList(nt),
                           SURV_FOREST.LeftNodeList(nt),
                           SURV_FOREST.RightNodeList(nt),
                           SURV_FOREST.NodeWeightList(nt));
        Find_Terminal_Node(0, OneTree, X, Ncat, proxy_id, real_id, TermNode);
        AllTermNode.col(nt) = TermNode;
      }

#pragma omp barrier

#pragma omp for schedule(static)
      for (size_t i = 0; i < N; i++) {
        mat pred_i_hazard_full(ntrees, NFail);
        for (size_t nt = 0; nt < ntrees; nt++) {
          arma::vec H_full = SURV_FOREST.NodeHazList(nt).at(AllTermNode(i, nt));
          pred_i_hazard_full.row(nt) = H_full.subvec(1, NFail).t();
        }

        mat pred_i_survival_full = cumprod(1 - pred_i_hazard_full, 1);

        mat pred_i_hazard_reduced = pred_i_hazard_full.cols(mapping);
        mat pred_i_survival_reduced = pred_i_survival_full.cols(mapping);

        Hazard.row(i) = mean(pred_i_hazard_reduced, 0);
        CHazard.row(i) = cumsum(Hazard.row(i));
        Surv.row(i) = exp(-CHazard.row(i));

        if (VarEst) {
          mat pred_i_for_var = cumsum(pred_i_hazard_reduced, 1);

          if (var_mode == 1) {
            size_t B = ntrees / 2;
            mat Diff =
                pred_i_for_var.rows(0, B - 1) - pred_i_for_var.rows(B, ntrees - 1);
            mat Vh = Diff.t() * Diff / ntrees;
            mat Vs = cov(pred_i_for_var, 1);
            mat diffmat = Vh - Vs;
            vec eigval;
            mat eigvec;
            eig_sym(eigval, eigvec, diffmat);
            eigval.elem(find(eigval < 1e-6)).fill(1e-6);
            Cov.slice(i) = eigvec * diagmat(eigval) * eigvec.t();
            Cov.slice(i) = (Cov.slice(i) + Cov.slice(i).t()) / 2;
          } else if (var_mode == 2) {
            Cov.slice(i) = compute_surv_ij_variance(pred_i_for_var, ObsTrack);
          } else if (var_mode == 3) {
            Cov.slice(i) = compute_surv_jack_variance(pred_i_for_var, ObsTrack);
          }
        }

        if (keep_all)
          AllHazard.slice(i).cols(0, effective_grid_size - 1) =
              pred_i_hazard_reduced;
      }
    }
  }

  py::dict out;
  out["Hazard"] = mat2np(Hazard);
  out["CHF"] = mat2np(CHazard);
  out["Survival"] = mat2np(Surv);
  if (VarEst) {
    // return per-subject layout: (N, T, T)
    py::array_t<double> cov({(py::ssize_t)N,
                             (py::ssize_t)effective_grid_size,
                             (py::ssize_t)effective_grid_size});
    auto w = cov.mutable_unchecked<3>();
    for (size_t i = 0; i < N; ++i)
      for (size_t a = 0; a < effective_grid_size; ++a)
        for (size_t b = 0; b < effective_grid_size; ++b)
          w(i, a, b) = Cov(a, b, i);
    out["Cov"] = cov;
  }
  if (keep_all) {
    py::array_t<double> all({(py::ssize_t)N, (py::ssize_t)ntrees,
                             (py::ssize_t)effective_grid_size});
    auto w = all.mutable_unchecked<3>();
    for (size_t i = 0; i < N; ++i)
      for (size_t nt = 0; nt < ntrees; ++nt)
        for (size_t t = 0; t < effective_grid_size; ++t)
          w(i, nt, t) = AllHazard(nt, t, i);
    out["AllHazard"] = all;
  }
  return out;
}

// ============================================================
// Utilities
// ============================================================

static double cindex_np(const dmat_t& y, const imat_t& censor,
                        const dmat_t& pred) {
  arma::vec Y = np2vec(y);
  arma::uvec Censor = np2uvec(censor);
  arma::vec p = np2vec(pred);
  return cindex_impl<arma::vec>(Y, Censor, p);
}

static py::array_t<int64_t> gen_ms_obs_track_mat(size_t ntrain, size_t k,
                                                 size_t ntrees, size_t seed) {
  size_t ntrees_half = ntrees / 2;
  arma::imat index_mat(ntrain, ntrees, arma::fill::zeros);
  Rand rng(seed);

  for (size_t i = 0; i < ntrees_half; i++) {
    arma::uvec rand_idx = rng.sample(0, ntrain - 1, 2 * k, 0);

    uvec left = {(uword)i};
    uvec right = {(uword)(i + ntrees_half)};

    index_mat.submat(rand_idx.head(k), left).fill(1);
    index_mat.submat(rand_idx.tail(k), right).fill(1);
  }
  return imat2np(index_mat);
}

// ============================================================
// Module
// ============================================================

PYBIND11_MODULE(_core, m) {
  m.doc() = "RLT C++ core: regression / classification / survival forests";

  py::class_<rlt::CoreParams>(m, "CoreParams")
      .def(py::init<>())
      .def_readwrite("n", &rlt::CoreParams::n)
      .def_readwrite("p", &rlt::CoreParams::p)
      .def_readwrite("ntrees", &rlt::CoreParams::ntrees)
      .def_readwrite("mtry", &rlt::CoreParams::mtry)
      .def_readwrite("nmin", &rlt::CoreParams::nmin)
      .def_readwrite("nsplit", &rlt::CoreParams::nsplit)
      .def_readwrite("replacement", &rlt::CoreParams::replacement)
      .def_readwrite("resample_prob", &rlt::CoreParams::resample_prob)
      .def_readwrite("use_obs_w", &rlt::CoreParams::use_obs_w)
      .def_readwrite("use_var_prob", &rlt::CoreParams::use_var_prob)
      .def_readwrite("importance", &rlt::CoreParams::importance)
      .def_readwrite("reinforcement", &rlt::CoreParams::reinforcement)
      .def_readwrite("obs_track", &rlt::CoreParams::obs_track)
      .def_readwrite("var_mode", &rlt::CoreParams::var_mode)
      .def_readwrite("linear_comb", &rlt::CoreParams::linear_comb)
      .def_readwrite("alpha", &rlt::CoreParams::alpha)
      .def_readwrite("split_rule", &rlt::CoreParams::split_rule)
      .def_readwrite("linear_comb_method", &rlt::CoreParams::linear_comb_method)
      .def_readwrite("embed_ntrees", &rlt::CoreParams::embed_ntrees)
      .def_readwrite("embed_mtry", &rlt::CoreParams::embed_mtry)
      .def_readwrite("embed_nmin", &rlt::CoreParams::embed_nmin)
      .def_readwrite("embed_nsplit", &rlt::CoreParams::embed_nsplit)
      .def_readwrite("embed_replacement", &rlt::CoreParams::embed_replacement)
      .def_readwrite("embed_resample_prob",
                     &rlt::CoreParams::embed_resample_prob)
      .def_readwrite("embed_mute", &rlt::CoreParams::embed_mute)
      .def_readwrite("embed_protect", &rlt::CoreParams::embed_protect)
      .def_readwrite("embed_threshold", &rlt::CoreParams::embed_threshold)
      .def_readwrite("ncores", &rlt::CoreParams::ncores)
      .def_readwrite("verbose", &rlt::CoreParams::verbose)
      .def_readwrite("seed", &rlt::CoreParams::seed);

  m.def("RegUniForestFit", &RegUniForestFit, py::arg("x"), py::arg("y"),
        py::arg("ncat"), py::arg("obsweight"), py::arg("varprob"),
        py::arg("obstrack"), py::arg("params"));

  m.def("RegUniForestPred", &RegUniForestPred, py::arg("splitvar"),
        py::arg("splitvalue"), py::arg("leftnode"), py::arg("rightnode"),
        py::arg("nodeweight"), py::arg("nodeave"), py::arg("x"),
        py::arg("ncat"), py::arg("obstrack"), py::arg("var_mode"),
        py::arg("keep_all"), py::arg("ncores"), py::arg("verbose"));

  m.def("ClaUniForestFit", &ClaUniForestFit, py::arg("x"), py::arg("y"),
        py::arg("ncat"), py::arg("nclass"), py::arg("obsweight"),
        py::arg("varprob"), py::arg("obstrack"), py::arg("params"));

  m.def("ClaUniForestPred", &ClaUniForestPred, py::arg("splitvar"),
        py::arg("splitvalue"), py::arg("leftnode"), py::arg("rightnode"),
        py::arg("nodeweight"), py::arg("nodeprob"), py::arg("x"),
        py::arg("ncat"), py::arg("obstrack"), py::arg("var_mode"),
        py::arg("keep_all"), py::arg("ncores"), py::arg("verbose"));

  m.def("SurvUniForestFit", &SurvUniForestFit, py::arg("x"), py::arg("y"),
        py::arg("censor"), py::arg("ncat"), py::arg("obsweight"),
        py::arg("varprob"), py::arg("obstrack"), py::arg("params"));

  m.def("SurvUniForestPred", &SurvUniForestPred, py::arg("splitvar"),
        py::arg("splitvalue"), py::arg("leftnode"), py::arg("rightnode"),
        py::arg("nodeweight"), py::arg("nodehaz"), py::arg("x"),
        py::arg("ncat"), py::arg("nfail"), py::arg("mapping_indices"),
        py::arg("obstrack"), py::arg("var_mode"), py::arg("keep_all"),
        py::arg("ncores"), py::arg("verbose"));

  m.def("RegUniCombForestFit", &RegUniCombForestFit, py::arg("x"),
        py::arg("y"), py::arg("ncat"), py::arg("obsweight"),
        py::arg("varprob"), py::arg("obstrack"), py::arg("params"));

  m.def("RegUniCombForestPred", &RegUniCombForestPred, py::arg("splitvar"),
        py::arg("splitload"), py::arg("splitvalue"), py::arg("leftnode"),
        py::arg("rightnode"), py::arg("nodeweight"), py::arg("nodeave"),
        py::arg("x"), py::arg("ncat"), py::arg("obstrack"),
        py::arg("var_mode"), py::arg("keep_all"), py::arg("ncores"),
        py::arg("verbose"));

  m.def("ClaUniCombForestFit", &ClaUniCombForestFit, py::arg("x"),
        py::arg("y"), py::arg("ncat"), py::arg("nclass"),
        py::arg("obsweight"), py::arg("varprob"), py::arg("obstrack"),
        py::arg("params"));

  m.def("ClaUniCombForestPred", &ClaUniCombForestPred, py::arg("splitvar"),
        py::arg("splitload"), py::arg("splitvalue"), py::arg("leftnode"),
        py::arg("rightnode"), py::arg("nodeweight"), py::arg("nodeprob"),
        py::arg("x"), py::arg("ncat"), py::arg("obstrack"),
        py::arg("var_mode"), py::arg("keep_all"), py::arg("ncores"),
        py::arg("verbose"));

  m.def("SurvUniCombForestFit", &SurvUniCombForestFit, py::arg("x"),
        py::arg("y"), py::arg("censor"), py::arg("ncat"),
        py::arg("obsweight"), py::arg("varprob"), py::arg("obstrack"),
        py::arg("params"));

  m.def("SurvUniCombForestPred", &SurvUniCombForestPred, py::arg("splitvar"),
        py::arg("splitload"), py::arg("splitvalue"), py::arg("leftnode"),
        py::arg("rightnode"), py::arg("nodeweight"), py::arg("nodehaz"),
        py::arg("x"), py::arg("ncat"), py::arg("nfail"),
        py::arg("mapping_indices"), py::arg("obstrack"), py::arg("var_mode"),
        py::arg("keep_all"), py::arg("ncores"), py::arg("verbose"));

  m.def("cindex", &cindex_np, py::arg("y"), py::arg("censor"),
        py::arg("pred"));

  m.def("gen_ms_obs_track_mat", &gen_ms_obs_track_mat, py::arg("ntrain"),
        py::arg("k"), py::arg("ntrees"), py::arg("seed"));
}
