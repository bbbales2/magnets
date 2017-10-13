library(tidyverse)
library(ggplot2)
library(rstan)
library(inline)
library(Rcpp)
library(RcppEigen)

sourceCpp("covariance.cpp")

(csv = read_csv('cooling_only_Gd_DSm.csv') %>% #'cooling_only_Gd_DSm.csv'
  rename(temp = `Temperature (K)`, field = `Magnetic Field (Oe)`, y = normalized_moment_cgs))

(csv = read_csv('FeGe_ders.csv') %>%
  rename(y = der))

csv %>% ggplot(aes(temp, y)) +
  geom_point(aes(colour = field), size = 0.1)

(df = csv %>%
  mutate(field = round(field, -1)) %>%
  group_by(field) %>% sample_n(50) %>% filter(field == 5000) %>%
  ungroup())

ltemp = 30.0
lfield = 10000.0
sigmaf = 100.0
N = nrow(df)

temp = df %>% pull(temp)
Sigma = sigmaf^2 * rbf_cov(temp, temp, ltemp)

tempp = seq(min(temp), max(temp), length = 1000)
Ks = sigmaf^2 * rbf_cov(tempp, temp, ltemp)

list(temp = tempp,
     y = (Ks %*% solve(Sigma, df %>% pull(y))) %>% as.vector) %>% as.tibble %>%
  ggplot(aes(temp, y)) +
  geom_point(size = 0.2) +
  geom_line() +
  geom_point(data = df, colour = "red", size = 0.1)

ltemp = 10.0
lfield = 1.0
sigmaf = 50.0
sigma = 5.0

(df = csv %>%
    mutate(field = round(field, 4)) %>%
    mutate(field = log(field)) %>%
    group_by(field) %>% sample_frac(0.2) %>%
    ungroup())

df %>% ggplot(aes(temp, y)) +
  geom_point(aes(colour = field), size = 0.1)

xdata = df %>% select(temp, field) %>% as.matrix
Sigma = sigmaf^2 * rbf_cov_vec(xdata, xdata, c(ltemp, lfield)) + diag(sigma, nrow(xdata))

temp = df %>% pull(temp)
field = df %>% pull(field)
tempp = seq(min(temp), max(temp), length = 50)
fieldp = log(seq(exp(min(field)), exp(max(field)), length = 50))
xinterp = expand.grid(tempp, fieldp) %>% as.matrix

Ks = sigmaf^2 * rbf_cov_vec(xinterp, xdata, c(ltemp, lfield))

Sigmainvy = fsolve(Sigma, df %>% pull(y))
#solve(Sigma, df %>% pull(y))

xinterp %>% as.tibble %>%
  rename(temp = Var1, field = Var2) %>%
  mutate(y = Ks %*% Sigmainvy) %>%
  ggplot(aes(exp(field), y)) +
  geom_point(aes(colour = temp), size = 0.1) +
  geom_point(data = df, size = 0.1, colour = "red")

xinterp %>% as.tibble %>%
  rename(temp = Var1, field = Var2) %>%
  mutate(y = Ks %*% Sigmainvy) %>%
  ggplot(aes(temp, y)) +
  geom_point(aes(colour = exp(field)), size = 0.1) +
  geom_point(data = df, size = 0.1, colour = "red")

xinterp %>% as.tibble %>%
  rename(temp = Var1, field = Var2) %>%
  mutate(y = Ks %*% Sigmainvy) %>%
  ggplot(aes(temp, exp(field))) +
  geom_tile(aes(fill = y), size = 0.1) +
  geom_point(data = df, size = 0.1, colour = "red")
