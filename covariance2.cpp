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

std::vector<int> which_int(const StringVector &string_which) {
  std::vector<int> which(string_which.size());
  for(int i = 0; i < string_which.size(); i++) {
    if(strcmp(string_which(i), "normal") == 0) {
      which[i] = 0;
    } else if(strcmp(string_which(i), "derivative") == 0) {
      which[i] = -1;
    } else if(strcmp(string_which(i), "integral") == 0) {
      which[i] = 1;
    } else {
      throw std::domain_error("string_which must be a character vector with "
                                "elements equal to normal, derivative, or integral");
    }
  }
  
  return which;
}

// [[Rcpp::export]]
NumericMatrix rbf_cov_w_vec(NumericMatrix x1, NumericMatrix x2,
                            NumericVector l, StringVector string_which1,
                            StringVector string_which2, double diag = 0.0) {
  std::vector<int> which1 = which_int(string_which1),
    which2 = which_int(string_which2);
  
  NumericMatrix Sigma(x1.nrow(), x2.nrow());
  
  if(x1.ncol() != x2.ncol() || x2.ncol() != l.size())
    throw std::domain_error("Cols of x1 should match cols of x2 should match size of l");
  
  for(int i = 0; i < x1.nrow(); i++) {
    for(int j = 0; j < x2.nrow(); j++) {
      double result = 1.0;
      
      for(int k = 0; k < l.size(); k++) {
        double rbf = std::exp(-(x1(i, k) - x2(j, k)) * (x1(i, k) - x2(j, k)) / (2 * l[k] * l[k]));
        if(which1[k] == -1 && which2[k] == -1) {
          result *= (l[k] + x1(i, k) - x2(j, k)) * (l[k] - x1(i, k) + x2(j, k)) * rbf / (l[k] * l[k] * l[k] * l[k]);
        } else if(which1[k] == -1 && which2[k] == 0) {
          result *= (x2(j, k) - x1(i, k)) * rbf / (l[k] * l[k]);
        } else if(which1[k] == -1 && which2[k] == 1) {
          throw std::invalid_argument("which1 == derivative, which2 == integral not implemented");
        } else if(which1[k] == 0 && which2[k] == -1) {
          result *= (x1(i, k) - x2(j, k)) * rbf / (l[k] * l[k]);
        } else if(which1[k] == 0 && which2[k] == 0) {
          result *= rbf;
        } else if(which1[k] == 0 && which2[k] == 1) {
          result *= l[k] * sqrt(M_PI / 2.0) * (erf((x1(i, k)) / (sqrt(2.0) * l[k])) + erf((x2(j, k) - x1(i, k)) / (sqrt(2.0) * l[k])));
        } else if(which1[k] == 1 && which2[k] == -1) {
          throw std::invalid_argument("which1 == integral, which2 == derivative not implemented");
        } else if(which1[k] == 1 && which2[k] == 0) {
          result *= l[k] * sqrt(M_PI / 2.0) * (erf((x2(j, k)) / (sqrt(2.0) * l[k])) + erf((x1(i, k) - x2(j, k)) / (sqrt(2.0) * l[k])));
        } else if(which1[k] == 1 && which2[k] == 1) {
          result *= l[k] * sqrt(M_PI / 2.0) * (
            l[k] * sqrt(2.0 / M_PI) * (
                -1 +
                  std::exp(-x1(i, k) * x1(i, k) / (2 * l[k] * l[k])) -
                  std::exp(-(x1(i, k) - x2(j, k)) * (x1(i, k) - x2(j, k)) / (2 * l[k] * l[k])) +
                  std::exp(-x2(j, k) * x2(j, k) / (2 * l[k] * l[k]))
            ) + 
              x1(i, k) * erf(x1(i, k) / (sqrt(2.0) * l[k])) +
              x2(j, k) * erf(x2(j, k) / (sqrt(2.0) * l[k])) +
              (x1(i, k) - x2(j, k)) * erf((x2(j, k) - x1(i, k)) / (sqrt(2.0) * l[k]))
          );
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
NumericMatrix fsolve(NumericMatrix X, NumericMatrix y) {
  typedef Eigen::Map<Eigen::MatrixXd> MapMatd;
  const MapMatd X_(as<MapMatd>(X));
  const MapMatd y_(as<MapMatd>(y));
  Eigen::MatrixXd out = X_.llt().solve(y_);
  return wrap(out);
}

//// [[Rcpp::export]]
/*NumericVector fvsolve(NumericMatrix X, NumericVector y) {
typedef Eigen::Map<Eigen::MatrixXd> MapMatd;
typedef Eigen::Map<Eigen::VectorXd> MapVecd;
const MapMatd X_(as<MapMatd>(X));
const MapVecd y_(as<MapVecd>(y));
Eigen::VectorXd out = X_.llt().solve(y_);
return wrap(out);
}*/

// [[Rcpp::export]]
List feigen(NumericMatrix X) {
  typedef Eigen::Map<Eigen::MatrixXd> MapMatd;
  const MapMatd X_(as<MapMatd>(X));
  Eigen::MatrixXd X__ = X_;
  //Eigen::VectorXd out = X_.llt().solve(y_);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es;
  
  es.compute(X__);
  
  List out;
  
  out["e"] = wrap(es.eigenvalues());
  out["v"] = wrap(es.eigenvalues());
  
  return wrap(out);
}