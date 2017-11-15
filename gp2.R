library(tidyverse)
library(ggplot2)
library(rstan)
library(pracma)
library(inline)
library(Rcpp)
library(RcppEigen)

sourceCpp("covariance2.cpp")
