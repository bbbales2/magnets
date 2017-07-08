library(tidyverse)
library(ggplot2)
library(rstan)
library(reshape2)
library(akima)
library(scales)
library(assist)

setwd("~/magnets")

df = read_csv("fege.csv") %>%
  rename(temp = `Temperature (K)`, mag = `Magnetic Field (T)`, cgs = normalized_moment_cgs) %>%
  mutate(mags = rescale(mag, c(0.0, 1.0))) %>%
  mutate(mags = round(mags, 2)) %>%
  mutate(temps = rescale(temp, c(0.0, 1.0))) %>%
  arrange(X1) %>%
  group_by(mags) %>%
#  filter(lead(temp) < temp) %>%
#  sample_n(100) %>%
  ungroup() %>%
  arrange(temps)
# & max(cgs) > 40
df %>% ggplot(aes(x = temp, y = cgs)) +
  geom_line(aes(color = mag, group = mags))

df %>% select(mags) %>% unique() %>% sample_n(1)

df2 = df %>% filter(mags == 0.4)

df2 %>% ggplot(aes(x = temp, y = cgs)) +
  geom_line()

fit = ssr(cgs ~ 1, data = df, rk = rk.prod(linear2(temps), linear2(mags)))
pred = predict(fit)$fit

N = 100
x = seq(0.0, 1.0, length = N)
dfp = as_tibble(list(x = x)) %>% mutate(idx = 1)
dfp = left_join(dfp, dfp, by = "idx") %>% rename(temps = x.x, mags = x.y) %>% select(-idx)
dfp$z = predict(fit, dfp)$fit

dfp %>% ggplot(aes(temps, mags, fill = z)) +
  geom_tile()

bind_cols(df, list(cgsp = pred)) %>% gather(which, value, c(cgs, cgsp)) %>% ggplot(aes(temp, value)) +
  geom_line(aes(group = which, colour = which, linetype = which)) +
  facet_wrap(~ mags)
