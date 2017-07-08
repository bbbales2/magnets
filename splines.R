library(tidyverse)
library(ggplot2)
library(rstan)
library(reshape2)
library(akima)
library(scales)
library(assist)

setwd("~/magnets")

df = read_csv("gd.csv") %>%
  rename(temp = `Temperature (K)`, mag = `Magnetic Field (T)`, cgs = normalized_moment_cgs) %>%
  mutate(mags = rescale(mag, c(0.0, 1.0))) %>%
  mutate(mags = round(mags, 2)) %>%
  mutate(temps = rescale(temp, c(0.0, 1.0))) %>%
  arrange(X1) %>%
  group_by(mags) %>%
  filter(lead(temp) < temp & max(cgs) > 40) %>%
  #sample_n(100) %>%
  ungroup()# %>%
#group_by(y) %>%
#mutate(rn = row_number()) %>%
#ungroup()# %>%
#

df %>% ggplot(aes(x = temp, y = cgs)) +
  geom_line(aes(color = mag, group = mags))

df %>% select(y) %>% unique() %>% sample_n(1)

df2 = df %>% filter(y == -0.1)

df2 %>% ggplot(aes(x = temp, y = cgs)) +
  geom_line()