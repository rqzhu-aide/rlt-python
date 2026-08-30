//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Tree Definitions
//  **********************************

// my header file

#include <armadillo>
#include "rlt_compat.h"
using namespace arma;

#include "Utility.h"

#ifndef RLT_TREE_DEFINITION
#define RLT_TREE_DEFINITION

// *********************** //
//  Node status codes     //
// *********************** //

enum NodeStatus : int { 
  Unused = -2,    // Available slot in tree array
  Reserved = -3,  // Temporarily reserved during tree building
  Terminal = -1   // Terminal node (leaf)
};

// *********************** //
//  Tree and forest class  //
// *********************** //

class Tree_Class{ // single split trees
public:
  arma::ivec& SplitVar;
  arma::vec& SplitValue;
  arma::uvec& LeftNode;
  arma::uvec& RightNode;
  arma::vec& NodeWeight;
  
  Tree_Class(arma::ivec& SplitVar,
             arma::vec& SplitValue,
             arma::uvec& LeftNode,
             arma::uvec& RightNode,
             arma::vec& NodeWeight) : SplitVar(SplitVar),
                                      SplitValue(SplitValue),
                                      LeftNode(LeftNode),
                                      RightNode(RightNode),
                                      NodeWeight(NodeWeight) {}
  
  void find_next_nodes(size_t& NextLeft, size_t& NextRight)
  {
    while( SplitVar(NextLeft) != NodeStatus::Unused ) NextLeft++;
    SplitVar(NextLeft) = NodeStatus::Reserved;  
    
    NextRight = NextLeft;
    while( SplitVar(NextRight) != NodeStatus::Unused ) NextRight++;
    
    SplitVar(NextRight) = NodeStatus::Reserved;
  }
  
  // get tree length
  size_t get_tree_length() {
    size_t i = 0;
    while (i < SplitVar.n_elem and SplitVar(i) != NodeStatus::Unused) i++;
    return( (i < SplitVar.n_elem) ? i:SplitVar.n_elem );
  }
  
  void print() {
    // Tree printing disabled in release build
  }
  
};

class Comb_Tree_Class{ // multivariate split trees
public:
  arma::imat& SplitVar;
  arma::mat& SplitLoad;
  arma::vec& SplitValue;
  arma::uvec& LeftNode;
  arma::uvec& RightNode;
  arma::vec& NodeWeight;
  
  Comb_Tree_Class(arma::imat& SplitVar,
                   arma::mat& SplitLoad,
                   arma::vec& SplitValue,
                   arma::uvec& LeftNode,
                   arma::uvec& RightNode,
                   arma::vec& NodeWeight) : SplitVar(SplitVar),
                                            SplitLoad(SplitLoad),
                                            SplitValue(SplitValue),
                                            LeftNode(LeftNode),
                                            RightNode(RightNode),
                                            NodeWeight(NodeWeight) {}
  
  void find_next_nodes(size_t& NextLeft, size_t& NextRight)
  {
    while( SplitVar(NextLeft, 0) != NodeStatus::Unused ) NextLeft++;
    SplitVar(NextLeft) = NodeStatus::Reserved;
    
    NextRight = NextLeft;
    
    while( SplitVar(NextRight, 0) != NodeStatus::Unused ) NextRight++;
    SplitVar(NextRight, 0) = NodeStatus::Reserved;
  }
  
  // get tree length
  size_t get_tree_length() {
    size_t i = 0;
    while (i < SplitVar.n_rows and SplitVar(i, 0) != NodeStatus::Unused) i++;
    return( (i < SplitVar.n_rows) ? i:SplitVar.n_rows );
  }
  
  void print() {
    // Tree printing disabled in release build
  }
  
};

// **************** //
// class for splits //
// **************** //

class Split_Class{ // univariate splits
public:
  size_t var = 0;
  double value = 0;
  double score = -1;
  
  void print(void) {
    // Split printing disabled in release build
  }
};

class Comb_Split_Class{ // multi-variate splits
public:
  arma::uvec& var;
  arma::vec& load;
  double value = 0;
  double score = -1;
  
  Comb_Split_Class(arma::uvec& var,
                   arma::vec& load) : 
    var(var),
    load(load) {}
  
  void print(void) {
    // Split printing disabled in release build
  }
};


// ************************ //
// for categorical variable //
// ************************ //

class Cat_Class{
public:
  
  virtual ~Cat_Class() = default;
  
  size_t cat = 0; // category number
  size_t count = 0; // count is used for setting nmin
  double weight = 0; // weight is used for calculation
  double score = 0; // for sorting
    
  void print() {
    // Category printing disabled in release build
  }
};

#endif
