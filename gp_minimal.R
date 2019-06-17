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
tempp = seq(min(temp), max(temp), length = 200)
fieldp = log(seq(exp(min(field)), exp(max(field)), length = 200))
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

x = matrix(dMdt_mean, nrow = length(tempp))
  
xint = matrix(0, nrow = nrow(x), ncol = ncol(x))

# Compute the integral of x (mean of dMdt) over the field
for(i in 2:length(fieldp)) {
  df = (exp(fieldp[i]) - exp(fieldp[i - 1]))
  xint[, i] = x[, i - 1] * df + xint[, i - 1]
}

mu_df = tibble(mu = xint %>% as.vector,
               temp = xinterp[, 1],
               field = xinterp[, 2])

mu_df %>%
    ggplot(aes(temp, exp(field))) +
    geom_tile(aes(fill = mu)) +
    scale_fill_gradientn(colours = rainbow(10))

