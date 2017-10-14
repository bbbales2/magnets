library(tidyverse)
library(ggplot2)
library(rstan)
library(inline)
library(Rcpp)
library(RcppEigen)

sourceCpp("covariance.cpp")

(csv = read_csv('FeGe_ders.csv') %>%
    rename(y = der))

ltemp = 0.5
lfield = 0.015
sigmaf = 1.0
sigma = 0.01

(df = csv %>%
    mutate(field = round(field, 4)) %>%
    group_by(field) %>% sample_frac(1.0) %>%
    ungroup())

df %>% ggplot(aes(temp, y)) +
  geom_point(aes(colour = field), size = 0.1)

xdata = df %>% select(temp, field) %>% as.matrix
Sigma = sigmaf^2 * rbf_cov_vec(xdata, xdata, c(ltemp, lfield)) + diag(sigma, nrow(xdata))

temp = df %>% pull(temp)
field = df %>% pull(field)
tempp = seq(min(temp), max(temp), length = 200)
fieldp = seq(min(field), max(field), length = 200)
xinterp = expand.grid(tempp, fieldp) %>% as.matrix

Ks = sigmaf^2 * rbf_cov_vec(xinterp, xdata, c(ltemp, lfield))

Sigmainvy = fsolve(Sigma, df %>% pull(y))
#solve(Sigma, df %>% pull(y))

xinterp %>% as.tibble %>%
  rename(temp = Var1, field = Var2) %>%
  mutate(y = Ks %*% Sigmainvy) %>%
  ggplot(aes(field, y)) +
  geom_point(aes(colour = temp), size = 0.1) +
  geom_point(data = df, size = 0.1, colour = "red")

xinterp %>% as.tibble %>%
  rename(temp = Var1, field = Var2) %>%
  mutate(y = Ks %*% Sigmainvy) %>%
  ggplot(aes(temp, y)) +
  geom_point(aes(colour = field), size = 0.1) +
  geom_point(data = df, size = 0.1, colour = "red")

xinterp %>% as.tibble %>%
  rename(temp = Var1, field = Var2) %>%
  mutate(y = Ks %*% Sigmainvy) %>%
  ggplot(aes(temp, field)) +
  geom_tile(aes(fill = y), size = 0.1) +
  geom_point(data = df, size = 0.1, colour = "red") +
  scale_fill_gradient2(low = "#0000FF", high = "#FF0000", mid = "white", midpoint = 0.0, limits = c(-.2, .2))