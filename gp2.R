library(MASS)
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
    group_by(field) %>% sample_frac(0.1) %>%
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
tempp = seq(min(temp), max(temp), length = 20)
fieldp = log(seq(exp(min(field)), exp(max(field)), length = 50))
#fieldp = seq(min(field), max(field), length = 200)
xinterp = expand.grid(tempp, fieldp) %>% as.matrix

Ks = sigmaf^2 * rbf_cov_w_vec(xinterp, xdata, c(ltemp, lfield),
                              c("normal", "normal"),
                              c("normal", "normal"))

Kds = sigmaf^2 * rbf_cov_w_vec(xinterp, xdata, c(ltemp, lfield),
                                c("derivative", "normal"),
                                c("normal", "normal"))

Kdsds = sigmaf^2 * rbf_cov_w_vec(xinterp, xinterp, c(ltemp, lfield),
                               c("derivative", "normal"),
                               c("derivative", "normal"), 1e-10)

#Kdis = sigmaf^2 * rbf_cov_w_vec(xinterp, xdata, c(ltemp, lfield),
#                               c("derivative", "integral"),
#                               c("normal", "normal"))

Sigmainvy = fsolve(Sigma, df %>% pull(y) %>% as.matrix)
dMdt_mean = Kds %*% Sigmainvy
dMdt_var = Kdsds - Kds %*% fsolve(Sigma, t(Kds))
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

LdMdt = chol(dMdt_var) %>% t

#x = dMdt %>% pull(y) %>%
#  matrix(nrow = length(tempp))

Ss = list()
for(r in 1:10) {
  x = (dMdt_mean + LdMdt %*% rnorm(length(dMdt_mean))) %>%
    matrix(nrow = length(tempp))
  
  xint = matrix(0, nrow = nrow(x), ncol = ncol(x))
  
  for(i in 2:length(fieldp)) {
    xint[, i] = (exp(fieldp[i]) - exp(fieldp[i - 1])) * x[, i - 1] + xint[, i - 1]
  }
  
  Ss[[r]] = list(S = xint %>% as.vector,
                 r = r,
                 temp = xinterp[, 1],
                 field = xinterp[, 2]) %>% as.tibble
}

Sdf = bind_rows(Ss) %>%
  group_by(temp, field) %>%
  summarize(mu = mean(S),
            ql = quantile(S, 0.25),
            qh = quantile(S, 0.75))

Sdf %>%
  filter(field %in% fieldp[c(10, 20, 30)]) %>%
  ggplot(aes(temp, mu)) +
  geom_line(aes(group = field, colour = field)) +
  geom_line(aes(temp, ql, group = field, colour = field), linetype = "dotted") +
  geom_line(aes(temp, qh, group = field, colour = field), linetype = "dotted")

# Sdf = xinterp %>% as.tibble %>%
#   rename(temp = Var1, field = Var2) %>%
#   mutate(S_mean = S_mean %>% as.vector,
#          S_var = S_var %>% as.vector)
# 
# Sdf %>%
#   ggplot(aes(exp(field), S_mean)) +
#   geom_point(aes(colour = temp), size = 0.1)
# 
# Sdf %>%
#   ggplot(aes(temp, S_mean)) +
#   geom_point(aes(colour = exp(field)), size = 0.1)
# 
# Sdf %>%
#   filter(temp > 250 & temp < 350) %>%
#   ggplot(aes(temp, exp(field))) +
#   geom_tile(aes(fill = S_mean)) +
#   scale_fill_gradientn(colours = rainbow(10))
# 
# Sdf %>%
#   filter(temp > 250 & temp < 350) %>%
#   ggplot(aes(temp, exp(field))) +
#   geom_tile(aes(fill = sqrt(S_var))) +
#   scale_fill_gradientn(colours = rainbow(10))
# 
# S %>% summarize_all(min)

