//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Survival
//  **********************************

// my header files
# include "../core/Tree_Definition.h"
# include "../core/Utility.h"
# include "../core/Tree_Function.h"
using namespace arma;

#ifndef RLT_SURV_UNI_DEFINITION
#define RLT_SURV_UNI_DEFINITION

// ************ //
//  data class  //
// ************ //

class RLT_SURV_DATA{
public:
  arma::mat& X;
  arma::uvec& Y;
  arma::uvec& Censor;
  arma::uvec& Ncat;
  arma::size_t& NFail;
  arma::vec& obsweight;
  arma::vec& varprob;
  
  RLT_SURV_DATA(arma::mat& X, 
               arma::uvec& Y,
               arma::uvec& Censor,
               arma::uvec& Ncat,
               arma::size_t& NFail,
               arma::vec& obsweight,
               arma::vec& varprob) : X(X), 
               Y(Y), 
               Censor(Censor),
               Ncat(Ncat), 
               NFail(NFail),
               obsweight(obsweight), 
               varprob(varprob) {}
};

// forest class survival 

class Surv_Uni_Forest_Class{
public:
  arma::field<arma::ivec>& SplitVarList;
  arma::field<arma::vec>& SplitValueList;
  arma::field<arma::uvec>& LeftNodeList;
  arma::field<arma::uvec>& RightNodeList;
  arma::field<arma::vec>& NodeWeightList;
  arma::field<arma::field<arma::vec>>& NodeHazList;
  
  Surv_Uni_Forest_Class(arma::field<arma::ivec>& SplitVarList,
                       arma::field<arma::vec>& SplitValueList,
                       arma::field<arma::uvec>& LeftNodeList,
                       arma::field<arma::uvec>& RightNodeList,
                       arma::field<arma::vec>& NodeWeightList,
                       arma::field<arma::field<arma::vec>>& NodeHazList) : SplitVarList(SplitVarList),
                                                                           SplitValueList(SplitValueList),
                                                                           LeftNodeList(LeftNodeList),
                                                                           RightNodeList(RightNodeList),
                                                                           NodeWeightList(NodeWeightList),
                                                                           NodeHazList(NodeHazList) {}
};

// tree class survival 

class Surv_Uni_Tree_Class : public Tree_Class{
public:
  arma::field<arma::vec>& NodeHaz;
  
  Surv_Uni_Tree_Class(arma::ivec& SplitVar,
                      arma::vec& SplitValue,
                      arma::uvec& LeftNode,
                      arma::uvec& RightNode,
                      arma::vec& NodeWeight,
                      arma::field<arma::vec>& NodeHaz) : Tree_Class(SplitVar,
                                                                    SplitValue,
                                                                    LeftNode,
                                                                    RightNode,
                                                                    NodeWeight),
                                                         NodeHaz(NodeHaz) {}

  // initiate tree
  void initiate(size_t TreeLength)
  {
    if (TreeLength == 0) TreeLength = 1;

    SplitVar.set_size(TreeLength);
    SplitVar.fill(NodeStatus::Unused);
    SplitVar(0) = NodeStatus::Reserved;

    SplitValue.zeros(TreeLength);
    LeftNode.zeros(TreeLength);
    RightNode.zeros(TreeLength);
    NodeWeight.zeros(TreeLength);
    NodeHaz.set_size(TreeLength);
  }

  // trim tree
  void trim(size_t TreeLength)
  {
    SplitVar.resize(TreeLength);
    SplitValue.resize(TreeLength);
    LeftNode.resize(TreeLength);
    RightNode.resize(TreeLength);
    NodeWeight.resize(TreeLength);
    field_vec_resize(NodeHaz, TreeLength);
  }

  // extend tree
  void extend()
  {
    // tree is not long enough, extend
    size_t OldLength = SplitVar.n_elem;
    size_t NewLength = (OldLength*1.5 > OldLength + 100)? (size_t) (OldLength*1.5):(OldLength + 100);

    SplitVar.resize(NewLength);
    SplitVar(span(OldLength, NewLength-1)).fill(NodeStatus::Unused);

    SplitValue.resize(NewLength);
    SplitValue(span(OldLength, NewLength-1)).zeros();

    LeftNode.resize(NewLength);
    LeftNode(span(OldLength, NewLength-1)).zeros();

    RightNode.resize(NewLength);
    RightNode(span(OldLength, NewLength-1)).zeros();

    NodeWeight.resize(NewLength);
    NodeWeight(span(OldLength, NewLength-1)).zeros();
    
    field_vec_resize(NodeHaz, NewLength);
  }
};

// #####################################
// ## Combination Split Forest/Tree  ##
// #####################################

class Surv_Uni_Comb_Forest_Class{
public:
  arma::field<arma::imat>& SplitVarList;
  arma::field<arma::mat>& SplitLoadList;
  arma::field<arma::vec>& SplitValueList;
  arma::field<arma::uvec>& LeftNodeList;
  arma::field<arma::uvec>& RightNodeList;
  arma::field<arma::vec>& NodeWeightList;
  arma::field<arma::field<arma::vec>>& NodeHazList;
  
  Surv_Uni_Comb_Forest_Class(arma::field<arma::imat>& SplitVarList,
                             arma::field<arma::mat>& SplitLoadList,
                             arma::field<arma::vec>& SplitValueList,
                             arma::field<arma::uvec>& LeftNodeList,
                             arma::field<arma::uvec>& RightNodeList,
                             arma::field<arma::vec>& NodeWeightList,
                             arma::field<arma::field<arma::vec>>& NodeHazList) : SplitVarList(SplitVarList),
                                                                                  SplitLoadList(SplitLoadList),
                                                                                  SplitValueList(SplitValueList),
                                                                                  LeftNodeList(LeftNodeList),
                                                                                  RightNodeList(RightNodeList),
                                                                                  NodeWeightList(NodeWeightList),
                                                                                  NodeHazList(NodeHazList) {}
};

class Surv_Uni_Comb_Tree_Class : public Comb_Tree_Class{
public:
  arma::field<arma::vec>& NodeHaz;
  
  Surv_Uni_Comb_Tree_Class(arma::imat& SplitVar,
                           arma::mat& SplitLoad,
                           arma::vec& SplitValue,
                           arma::uvec& LeftNode,
                           arma::uvec& RightNode,
                           arma::vec& NodeWeight,
                           arma::field<arma::vec>& NodeHaz) : Comb_Tree_Class(SplitVar,
                                                                              SplitLoad,
                                                                              SplitValue,
                                                                              LeftNode,
                                                                              RightNode,
                                                                              NodeWeight),
                                                              NodeHaz(NodeHaz) {}

  // initiate tree
  void initiate(size_t TreeLength, size_t linear_comb)
  {
    if (TreeLength == 0) TreeLength = 1;
    
    SplitVar.zeros(TreeLength, linear_comb);
    SplitVar.col(0).fill(NodeStatus::Unused);
    
    SplitLoad.zeros(TreeLength, linear_comb);
    
    SplitValue.zeros(TreeLength);
    LeftNode.zeros(TreeLength);
    RightNode.zeros(TreeLength);
    NodeWeight.zeros(TreeLength);
    NodeHaz.set_size(TreeLength);
  }

  // trim tree
  void trim(size_t TreeLength)
  {
    SplitVar.resize(TreeLength, SplitVar.n_cols);
    SplitLoad.resize(TreeLength, SplitLoad.n_cols);
    SplitValue.resize(TreeLength);
    LeftNode.resize(TreeLength);
    RightNode.resize(TreeLength);
    NodeWeight.resize(TreeLength);
    field_vec_resize(NodeHaz, TreeLength);
  }

  // extend tree
  void extend()
  {
    // tree is not long enough, extend
    size_t OldLength = SplitVar.n_rows;
    size_t NewLength = (OldLength*1.5 > OldLength + 100)? (size_t) (OldLength*1.5):(OldLength + 100);
    
    SplitVar.resize(NewLength, SplitVar.n_cols);
    SplitVar.rows(OldLength, NewLength-1).zeros();
    SplitVar.submat(OldLength, 0, NewLength-1, 0).fill(NodeStatus::Unused);
    
    SplitLoad.resize(NewLength, SplitLoad.n_cols);
    SplitLoad.rows(OldLength, NewLength-1).zeros();
    
    SplitValue.resize(NewLength);
    SplitValue.subvec(OldLength, NewLength-1).zeros();
    
    LeftNode.resize(NewLength);
    LeftNode.subvec(OldLength, NewLength-1).zeros();
    
    RightNode.resize(NewLength);
    RightNode.subvec(OldLength, NewLength-1).zeros();

    NodeWeight.resize(NewLength);
    NodeWeight.subvec(OldLength, NewLength-1).zeros();
    
    field_vec_resize(NodeHaz, NewLength);
  }
};

class Surv_Cat_Class: public Cat_Class{
public:
  arma::uvec FailCount;
  arma::uvec RiskCount;
  arma::vec WFailCount;
  arma::vec WRiskCount;
  arma::vec W2RiskCount;
  size_t nfail; 
  
  void initiate(size_t j, size_t NFail)
  {
    cat = j;
    nfail = 0;
    FailCount.zeros(NFail+1);
    RiskCount.zeros(NFail+1);
    WFailCount.zeros(NFail+1);
    WRiskCount.zeros(NFail+1);
    W2RiskCount.zeros(NFail+1);
  }
  
  void print(void) {
  }
  
};

#endif
