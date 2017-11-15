library(tidyverse)
library(ggplot2)
library(rstan)
library(pracma)
library(inline)
library(Rcpp)
library(RcppEigen)

sourceCpp("covariance2.cpp")

N = 100
l = 0.5
sigma = 0.1

x = matrix(1:N, nrow = N, 1) / N
y = x^2 + rnorm(N, 0, sigma)

UU = rbf_cov_w_vec(x, x, c(l), c("normal"), c("normal"), 1e-10)
UD = rbf_cov_w_vec(x, x, c(l), c("normal"), c("derivative"))
DU = rbf_cov_w_vec(x, x, c(l), c("derivative"), c("normal"))
DD = rbf_cov_w_vec(x, x, c(l), c("derivative"), c("derivative"))
UI = rbf_cov_w_vec(x, x, c(l), c("normal"), c("integral"))
II = rbf_cov_w_vec(x, x, c(l), c("integral"), c("integral"))

list(x = x %>% as.vector, y = y %>% as.vector) %>%
  as.tibble %>%
  ggplot(aes(x, y)) +
  geom_ribbon(aes(ymin = x^2 - 2 * sigma, ymax = x^2 + 2 * sigma), alpha = 0.25, fill = 'green') +
  geom_point() +
  geom_line(aes(x, x^2), color = 'red')

list(x = x %>% as.vector,
     y = t(UD) %*% fsolve(UU + diag(sigma^2, N), y) %>% as.vector,
     var = (DD - t(UD) %*% fsolve(UU + diag(sigma^2, N), UD)) %>% diag) %>%
  as.tibble %>%
  ggplot(aes(x, y)) +
  geom_ribbon(aes(ymin = y - 2 * sqrt(var), ymax = y + 2 * sqrt(var)), alpha = 0.25) +
  geom_line() +
  geom_line(aes(x, 2 * x), color = 'red')

list(x = as.vector(x),
     y = as.vector(t(UI) %*% fsolve(UU + diag(sigma^2, N), y)),
     var = (II - t(UI) %*% fsolve(UU + diag(sigma^2, N), UI)) %>% diag) %>%
  as.tibble %>%
  ggplot(aes(x, y)) +
  geom_ribbon(aes(ymin = y - 2 * sqrt(var), ymax = y + 2 * sqrt(var)), alpha = 0.25) +
  geom_line() +
  geom_line(aes(x, x^3 / 3.0), color = 'red')
