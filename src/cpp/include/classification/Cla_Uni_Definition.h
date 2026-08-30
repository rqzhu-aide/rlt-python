//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Classification
//  **********************************

// my header files
# include "../core/Tree_Definition.h"
# include "../core/Utility.h"
# include "../core/Tree_Function.h"
using namespace arma;

#ifndef RLT_CLA_UNI_DEFINITION
#define RLT_CLA_UNI_DEFINITION

// ************ //
//  data class  //
// ************ //

class RLT_CLA_DATA{
public:
  arma::mat& X;
  arma::uvec& Y;
  arma::uvec& Ncat;
  size_t nclass;
  arma::vec& obsweight;
  arma::vec& varprob;
  
  RLT_CLA_DATA(arma::mat& X, 
               arma::uvec& Y,
               arma::uvec& Ncat,
               size_t nclass,
               arma::vec& obsweight,
               arma::vec& varprob) : X(X), 
                                       Y(Y), 
                                       Ncat(Ncat), 
                                       nclass(nclass),
                                       obsweight(obsweight), 
                                       varprob(varprob) {}
};

// forest class classification 

class Cla_Uni_Forest_Class{
public:
  arma::field<arma::ivec>& SplitVarList;
  arma::field<arma::vec>& SplitValueList;
  arma::field<arma::uvec>& LeftNodeList;
  arma::field<arma::uvec>& RightNodeList;
  arma::field<arma::vec>& NodeWeightList;
  arma::field<arma::mat>& NodeProbList;
  
  Cla_Uni_Forest_Class(arma::field<arma::ivec>& SplitVarList,
                       arma::field<arma::vec>& SplitValueList,
                       arma::field<arma::uvec>& LeftNodeList,
                       arma::field<arma::uvec>& RightNodeList,
                       arma::field<arma::vec>& NodeWeightList,
                       arma::field<arma::mat>& NodeProbList) : SplitVarList(SplitVarList), 
                                                               SplitValueList(SplitValueList),
                                                               LeftNodeList(LeftNodeList),
                                                               RightNodeList(RightNodeList),
                                                               NodeWeightList(NodeWeightList),
                                                               NodeProbList(NodeProbList) {}
};

class Cla_Uni_Tree_Class : public Tree_Class{
public:
  arma::mat& NodeProb;

  Cla_Uni_Tree_Class(arma::ivec& SplitVar,
                     arma::vec& SplitValue,
                     arma::uvec& LeftNode,
                     arma::uvec& RightNode,
                     arma::vec& NodeWeight,
                     arma::mat& NodeProb) : Tree_Class(SplitVar,
                                                      SplitValue,
                                                      LeftNode,
                                                      RightNode,
                                                      NodeWeight),
                                            NodeProb(NodeProb) {}

  // initiate tree
  void initiate(size_t TreeLength, size_t nclass)
  {
    if (TreeLength == 0) TreeLength = 1;

    SplitVar.set_size(TreeLength);
    SplitVar.fill(NodeStatus::Unused);
    SplitVar(0) = NodeStatus::Reserved;

    SplitValue.zeros(TreeLength);
    LeftNode.zeros(TreeLength);
    RightNode.zeros(TreeLength);
    NodeWeight.zeros(TreeLength);
    NodeProb.zeros(TreeLength, nclass);
  }

  // trim tree
  void trim(size_t TreeLength)
  {
    SplitVar.resize(TreeLength);
    SplitValue.resize(TreeLength);
    LeftNode.resize(TreeLength);
    RightNode.resize(TreeLength);
    NodeWeight.resize(TreeLength);
    NodeProb.resize(TreeLength, NodeProb.n_cols);
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
    
    NodeProb.resize(NewLength, NodeProb.n_cols);
    NodeProb.rows(OldLength, NewLength-1).zeros();
  }
};


class Cla_Uni_Comb_Forest_Class{
public:
  arma::field<arma::imat>& SplitVarList;
  arma::field<arma::mat>& SplitLoadList;
  arma::field<arma::vec>& SplitValueList;
  arma::field<arma::uvec>& LeftNodeList;
  arma::field<arma::uvec>& RightNodeList;
  arma::field<arma::vec>& NodeWeightList;
  arma::field<arma::mat>& NodeProbList;
  
  Cla_Uni_Comb_Forest_Class(arma::field<arma::imat>& SplitVarList,
                            arma::field<arma::mat>& SplitLoadList,
                            arma::field<arma::vec>& SplitValueList,
                            arma::field<arma::uvec>& LeftNodeList,
                            arma::field<arma::uvec>& RightNodeList,
                            arma::field<arma::vec>& NodeWeightList,
                            arma::field<arma::mat>& NodeProbList) : SplitVarList(SplitVarList), 
                                                                    SplitLoadList(SplitLoadList),
                                                                    SplitValueList(SplitValueList),
                                                                    LeftNodeList(LeftNodeList),
                                                                    RightNodeList(RightNodeList),
                                                                    NodeWeightList(NodeWeightList),
                                                                    NodeProbList(NodeProbList) {}
};

class Cla_Uni_Comb_Tree_Class : public Comb_Tree_Class{
public:
  arma::mat& NodeProb;

  Cla_Uni_Comb_Tree_Class(arma::imat& SplitVar,
                          arma::mat& SplitLoad,
                          arma::vec& SplitValue,
                          arma::uvec& LeftNode,
                          arma::uvec& RightNode,
                          arma::vec& NodeWeight,
                          arma::mat& NodeProb) : Comb_Tree_Class(SplitVar,
                                                 SplitLoad,
                                                 SplitValue,
                                                 LeftNode,
                                                 RightNode,
                                                 NodeWeight),
                                                 NodeProb(NodeProb) {}
  
  // initiate tree
  void initiate(size_t TreeLength, size_t nclass, size_t linear_comb)
  {
    if (TreeLength == 0) TreeLength = 1;
    SplitVar.zeros(TreeLength, linear_comb);
    SplitVar.col(0).fill(NodeStatus::Unused);
    SplitVar(0) = NodeStatus::Reserved;

    SplitLoad.zeros(TreeLength, linear_comb);
    
    SplitValue.zeros(TreeLength);
    LeftNode.zeros(TreeLength);
    RightNode.zeros(TreeLength);
    NodeWeight.zeros(TreeLength);
    NodeProb.zeros(TreeLength, nclass);
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
    NodeProb.resize(TreeLength, NodeProb.n_cols);
  }
  
  // extend tree
  void extend()
  {
    // tree is not long enough, extend
    size_t OldLength = SplitVar.n_elem;
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
    
    NodeProb.resize(NewLength, NodeProb.n_cols);
    NodeProb.rows(OldLength, NewLength-1).zeros();
  }
};


class Cla_Cat_Class: public Cat_Class{
public:
  arma::vec Prob;
  
  void initiate(size_t j, size_t nclass)
  {
    cat = j;
    Prob.zeros(nclass); 
  }
  
  void print(void) {
  }
};

#endif
