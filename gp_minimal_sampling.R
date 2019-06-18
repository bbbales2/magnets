library(MASS)
library(tidyverse)
library(ggplot2)
library(rstan)
library(pracma)
library(inline)
library(Rcpp)
library(RcppEigen)

sourceCpp("covariance_minimal.cpp")

(csv = read_csv('MnB_temp_field_moment.csv') %>%
    rename(temp = `Temperature (K)`, field = `Magnetic Field (Oe)`, y = normalized_moment_cgs) %>%
    mutate(field = field / 10000.0,
           y = y) %>%
    filter(temp > min(temp) & temp < max(temp)))

# Round the field numbers so that we can group them later
df = csv %>%
  mutate(field = round(field, 4)) %>%
  mutate(field = log(field)) %>%
  group_by(field) %>% sample_frac(0.2) %>%
  ungroup()

ltemp = 10.0 # Length scale in temp direction
lfield = 1.0 # Length scale in field direction
sigmaf = 10.0 # Scale of y (how far the function deviates from zero) -- maybe use sd(df %>% pull(y))?
sigma = 0.01 # Observational noise, I assume you have an idea of this?

# Coordinates of measurements
xdata = df %>% select(temp, field) %>% as.matrix

# Covariance matrix for observations
Sigma = sigmaf^2 * rbf_cov_vec(xdata, xdata,
                               c(ltemp, lfield),
                               1e-10) +
  diag(sigma, nrow(xdata))

# Computer coordinates on which we will interpolate
temp = df %>% pull(temp)
field = df %>% pull(field)
tempp = seq(min(temp), max(temp), length = 50)
fieldp = log(seq(exp(min(field)), exp(max(field)), length = 100))
xinterp = expand.grid(tempp, fieldp) %>% as.matrix

# Covariance matrix between interpolation and observation points
Ks = sigmaf^2 * rbf_cov_vec(xinterp, xdata, c(ltemp, lfield))

# Covariance matrix between derivatives at interpolation points and observations
Kds = sigmaf^2 * rbf_cov_deriv_vec(xinterp, xdata, c(ltemp, lfield))

y_mean = df %>%
  pull(y) %>%
  mean

# http://www.gaussianprocess.org/gpml/chapters/RW2.pdf, eq. 2.19
# This is K(X, X)^{-1} (f - mean(f))
Sigmainvy = fsolve(Sigma, (df %>% pull(y) - y_mean) %>% as.matrix)

# Compute mean(f) + K(X_{*}, X) K(X, X)^{-1} f, where K is the covariance between
# M at observation and interpolation points
M_mean = y_mean + Ks %*% Sigmainvy

M = xinterp %>% as_tibble() %>%
  rename(temp = Var1, field = Var2) %>%
  mutate(y = M_mean)

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

# Compute K(X_{*}, X) K(X, X)^{-1} f, where K is the covariance between
# M at observation points and dMdtemp at interpolation points
dMdt_mean = Kds %*% Sigmainvy

dMdt = xinterp %>%
  as_tibble() %>%
  rename(temp = Var1, field = Var2) %>%
  mutate(y = dMdt_mean)

dMdt %>%
  ggplot(aes(exp(field), y)) +
  geom_point(aes(colour = temp), size = 0.1)

dMdt %>%
  ggplot(aes(temp, y)) +
  geom_point(aes(colour = exp(field)), size = 0.1)

dMdt %>%
  ggplot(aes(temp, exp(field))) +
  geom_tile(aes(fill = y))

Kdsds = sigmaf^2 * rbf_cov_deriv_deriv_vec(xinterp, xinterp, c(ltemp, lfield), 1e-10)
dMdt_mean = Kds %*% Sigmainvy
dMdt_var = Kdsds - Kds %*% fsolve(Sigma, t(Kds))

LdMdt = chol(dMdt_var) %>% t

Ss = list()
for(r in 1:100) {
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

samples_df = bind_rows(Ss) %>%
  group_by(temp, field) %>%
  summarize(mu = mean(S),
            ql = quantile(S, 0.25),
            qh = quantile(S, 0.75))

samples_df %>%
  filter(field %in% fieldp[c(10, 20, 30)]) %>%
  ggplot(aes(temp, mu)) +
  geom_line(aes(group = field, colour = field)) +
  geom_line(aes(temp, ql, group = field, colour = field), linetype = "dotted") +
  geom_line(aes(temp, qh, group = field, colour = field), linetype = "dotted")

samples_df %>%
  ggplot(aes(temp, exp(field))) +
  geom_tile(aes(fill = mu)) +
  scale_fill_gradientn(colours = rainbow(10))
