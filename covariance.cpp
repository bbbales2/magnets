#include <Rcpp.h>
#include <RcppEigen.h>
#include <cmath>
using namespace Rcpp;

// [[Rcpp::depends(RcppEigen)]]

// This is a simple example of exporting a C++ function to R. You can
// source this function into an R session using the Rcpp::sourceCpp 
// function (or via the Source button on the editor toolbar). Learn
// more about Rcpp at:
//
//   http://www.rcpp.org/
//   http://adv-r.had.co.nz/Rcpp.html
//   http://gallery.rcpp.org/
//

// [[Rcpp::export]]
NumericMatrix rbf_cov(NumericVector x1, NumericVector x2, double l) {
  NumericMatrix Sigma(x1.size(), x2.size());
  
  for(int i = 0; i < x1.size(); i++) {
    for(int j = 0; j < x2.size(); j++) {
      Sigma(i, j) = std::exp(-(x1[i] - x2[j]) * (x1[i] - x2[j]) / (2 * l * l));
    }
  }
  
  if(x1.size() == x2.size()) {
    for(int i = 0; i < x1.size(); i++) {
      Sigma(i, i) += 1e-10;
    }
  }

  return Sigma;
}

// [[Rcpp::export]]
NumericMatrix rbf_cov_vec(NumericMatrix x1, NumericMatrix x2, NumericVector l) {
  NumericMatrix Sigma(x1.nrow(), x2.nrow());
  
  if(x1.ncol() != x2.ncol() || x2.ncol() != l.size())
    throw std::domain_error("Cols of x1 should match cols of x2 should match size of l");
  
  for(int i = 0; i < x1.nrow(); i++) {
    for(int j = 0; j < x2.nrow(); j++) {
      double dot = 0.0;
      
      for(int k = 0; k < l.size(); k++) {
        dot += (x1(i, k) - x2(j, k)) * (x1(i, k) - x2(j, k)) / (2 * l[k] * l[k]);
      }
      
      Sigma(i, j) = std::exp(-dot);
    }
  }
  
  if(x1.nrow() == x2.nrow()) {
    for(int i = 0; i < x1.nrow(); i++) {
      Sigma(i, i) += 1e-10;
    }
  }
  
  return Sigma;
}

// [[Rcpp::export]]
NumericMatrix rbf_cov_d_vec(NumericMatrix x1, NumericMatrix x2, NumericVector l, int which) {
  NumericMatrix Sigma(x1.nrow(), x2.nrow());
  
  if(x1.ncol() != x2.ncol() || x2.ncol() != l.size())
    throw std::domain_error("Cols of x1 should match cols of x2 should match size of l");
  
  for(int i = 0; i < x1.nrow(); i++) {
    for(int j = 0; j < x2.nrow(); j++) {
      double dot = 0.0;
      double pre = -2 * (x1(i, which) - x2(j, which)) / (2 * l[which] * l[which]);
      
      for(int k = 0; k < l.size(); k++) {
        dot += (x1(i, k) - x2(j, k)) * (x1(i, k) - x2(j, k)) / (2 * l[k] * l[k]);
      }
      
      Sigma(i, j) = pre * std::exp(-dot);
    }
  }
  
  if(x1.nrow() == x2.nrow()) {
    for(int i = 0; i < x1.nrow(); i++) {
      Sigma(i, i) += 1e-10;
    }
  }
  
  return Sigma;
}

// [[Rcpp::export]]
NumericMatrix rbf_cov_i_vec(NumericMatrix x1, NumericMatrix x2, NumericVector l, int which) {
  NumericMatrix Sigma(x1.nrow(), x2.nrow());
  
  if(x1.ncol() != x2.ncol() || x2.ncol() != l.size())
    throw std::domain_error("Cols of x1 should match cols of x2 should match size of l");
  
  for(int i = 0; i < x1.nrow(); i++) {
    for(int j = 0; j < x2.nrow(); j++) {
      double sum = std::sqrt(3.14159265359 / 2.0) * l[which] * (erf((x1(i, which) - x2(i, which)) / (std::sqrt(2) * l[which])) + erf(x2(i, which) / (std::sqrt(2) * l[which])));

      for(int k = 0; k < l.size(); k++) {
        if(k != which) {
          sum += std::exp(-(x1(i, k) - x2(j, k)) * (x1(i, k) - x2(j, k)) / (2 * l[k] * l[k]));
        }
      }
      
      Sigma(i, j) = sum;
    }
  }
  
  if(x1.nrow() == x2.nrow()) {
    for(int i = 0; i < x1.nrow(); i++) {
      Sigma(i, i) += 1e-10;
    }
  }
  
  return Sigma;
}

// [[Rcpp::export]]
NumericVector fsolve(NumericMatrix X, NumericVector y) {
  typedef Eigen::Map<Eigen::MatrixXd> MapMatd;
  typedef Eigen::Map<Eigen::VectorXd> MapVecd;
  const MapMatd X_(as<MapMatd>(X));
  const MapVecd y_(as<MapVecd>(y));
  Eigen::VectorXd out = X_.llt().solve(y_);
  return wrap(out);
}
