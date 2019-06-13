#include <Rcpp.h>
#include <RcppEigen.h>
#include <cmath>
using namespace Rcpp;

// [[Rcpp::depends(RcppEigen)]]

// [[Rcpp::export]]
NumericMatrix rbf_cov_vec(NumericMatrix x1, NumericMatrix x2,
			  NumericVector l, double diag = 0.0) {
  NumericMatrix Sigma(x1.nrow(), x2.nrow());
  
  if(x1.ncol() != x2.ncol())
    throw std::domain_error("Cols of x1 should match cols of x2 should match size of l");

  if(x2.ncol() != l.size())
    throw std::domain_error("Cols of x1 and cols of x2 should match size of l");
  
  for(int i = 0; i < x1.nrow(); i++) {
    for(int j = 0; j < x2.nrow(); j++) {
      double result = 1.0;
      
      for(int k = 0; k < l.size(); k++) {
        double rbf = std::exp(-(x1(i, k) - x2(j, k)) * (x1(i, k) - x2(j, k)) / (2 * l[k] * l[k]));
	result *= rbf;
      }
      
      Sigma(i, j) = result;
    }
  }
  
  if(x1.nrow() == x2.nrow()) {
    for(int i = 0; i < x1.nrow(); i++) {
      Sigma(i, i) += diag;
    }
  }
  
  return Sigma;
}

// Check http://www.gaussianprocess.org/gpml/chapters/RW9.pdf section 9.4 "Derivative Observations"
// for how this function is computed

// [[Rcpp::export]]
NumericMatrix rbf_cov_deriv_vec(NumericMatrix x1, NumericMatrix x2,
				NumericVector l, double diag = 0.0) {
  NumericMatrix Sigma(x1.nrow(), x2.nrow());
  bool do_derivative[2] = { true, false };

  if(x1.ncol() != 2) {
    throw std::domain_error("This function is hardcoded to work only with 2 dimensional inputs. x1 and x2 must have 2 columns, and l must be length 2");
  }
  
  if(x1.ncol() != x2.ncol())
    throw std::domain_error("Cols of x1 should match cols of x2 should match size of l");

  if(x2.ncol() != l.size())
    throw std::domain_error("Cols of x1 and cols of x2 should match size of l");
  
  for(int i = 0; i < x1.nrow(); i++) {
    for(int j = 0; j < x2.nrow(); j++) {
      double result = 1.0;
      
      for(int k = 0; k < l.size(); k++) {
        double rbf = std::exp(-(x1(i, k) - x2(j, k)) * (x1(i, k) - x2(j, k)) / (2 * l[k] * l[k]));
        if(do_derivative[k] == true) {
          result *= (x2(j, k) - x1(i, k)) * rbf / (l[k] * l[k]);
        } else if(do_derivative[k] == false) {
          result *= rbf;
        }
      }
      
      Sigma(i, j) = result;
    }
  }
  
  if(x1.nrow() == x2.nrow()) {
    for(int i = 0; i < x1.nrow(); i++) {
      Sigma(i, i) += diag;
    }
  }
  
  return Sigma;
}

// [[Rcpp::export]]
NumericMatrix fsolve(NumericMatrix X, NumericMatrix y) {
  typedef Eigen::Map<Eigen::MatrixXd> MapMatd;
  const MapMatd X_(as<MapMatd>(X));
  const MapMatd y_(as<MapMatd>(y));
  Eigen::MatrixXd out = X_.llt().solve(y_);
  return wrap(out);
}
