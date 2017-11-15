library(tidyverse)
library(ggplot2)
library(rstan)
library(pracma)
library(inline)
library(Rcpp)
library(RcppEigen)

sourceCpp("covariance2.cpp")

(csv = read_csv('cooling_only_Gd_DSm.csv') %>% #'cooling_only_Gd_DSm.csv'
    rename(temp = `Temperature (K)`, field = `Magnetic Field (Oe)`, y = normalized_moment_cgs) %>%
    mutate(field = field / 10000.0,
           y = y) %>%
    filter(temp > min(temp) & temp < max(temp)))

csv %>% ggplot(aes(temp, y)) +
  geom_point(aes(colour = field), size = 0.1)

ltemp = 20.0
lfield = 1.0
sigmaf = 50.0
sigma = 5.0

(df = csv %>%
    mutate(field = round(field, 4)) %>%
    mutate(field = log(field)) %>%
    group_by(field) %>% sample_frac(0.05) %>%
    ungroup())

df %>% ggplot(aes(temp, y)) +
  geom_point(aes(colour = field), size = 0.1)

xdata = df %>% select(temp, field) %>% as.matrix
Sigma = sigmaf^2 * rbf_cov_w_vec(xdata, xdata, c(ltemp, lfield),
                                 c("normal", "normal"),
                                 c("normal", "normal"), 1e-10) +
  diag(sigma, nrow(xdata))

temp = df %>% pull(temp)
field = df %>% pull(field)
tempp = seq(min(temp), max(temp), length = 100)
fieldp = log(seq(exp(min(field)), exp(max(field)), length = 200))
#fieldp = seq(min(field), max(field), length = 200)
xinterp = expand.grid(tempp, fieldp) %>% as.matrix

Ks = sigmaf^2 * rbf_cov_w_vec(xinterp, xdata, c(ltemp, lfield),
                              c("normal", "normal"),
                              c("normal", "normal"))

Kdis = sigmaf^2 * rbf_cov_w_vec(xinterp, xdata, c(ltemp, lfield),
                               c("derivative", "integral"),
                               c("normal", "normal"))

Sigmainvy = fsolve(Sigma, df %>% pull(y) %>% as.matrix)
#solve(Sigma, df %>% pull(y))

M = xinterp %>% as.tibble %>%
  rename(temp = Var1, field = Var2) %>%
  mutate(y = Ks %*% Sigmainvy)

M %>%
  ggplot(aes(exp(field), y)) +
  geom_point(aes(colour = temp), size = 0.1) +
  geom_point(data = df, size = 0.1, colour = "red")

M %>%
  ggplot(aes(temp, y)) +
  geom_point(aes(colour = exp(field)), size = 0.1) +
  geom_point(data = df, size = 0.1, colour = "red")

M %>%
  ggplot(aes(temp, exp(field))) +
  geom_tile(aes(fill = y), size = 0.1) +
  geom_point(data = df, size = 0.1, colour = "red")

dMdt = xinterp %>% as.tibble %>%
  rename(temp = Var1, field = Var2) %>%
  mutate(y = Kds %*% Sigmainvy)

dMdt %>%
  ggplot(aes(exp(field), y)) +
  geom_point(aes(colour = temp), size = 0.1)

dMdt %>%
  ggplot(aes(temp, y)) +
  geom_point(aes(colour = exp(field)), size = 0.1)

dMdt %>%
  ggplot(aes(temp, exp(field))) +
  geom_tile(aes(fill = y))

S = xinterp %>% as.tibble %>%
  rename(temp = Var1, field = Var2) %>%
  mutate(y = Kdis %*% Sigmainvy)

S %>%
  ggplot(aes(exp(field), y)) +
  geom_point(aes(colour = temp), size = 0.1)

S %>%
  ggplot(aes(temp, y)) +
  geom_point(aes(colour = exp(field)), size = 0.1)

S %>%
  ggplot(aes(temp, exp(field))) +
  geom_tile(aes(fill = y)) +
  scale_fill_gradientn(colours = rainbow(10)) 

S %>% summarize_all(min)

