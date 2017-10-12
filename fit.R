library(tidyverse)
library(ggplot2)
library(rstan)
library(reshape2)
library(akima)
library(scales)

setwd("~/magnets")

df = read_csv("Gd_alldata.csv") %>%
  rename(temp = `Temperature (K)`, mag = `Magnetic Field (T)`, cgs = normalized_moment_cgs) %>%
  mutate(y = rescale(mag, c(-0.5, 0.5))) %>%
  mutate(y = round(y, 2)) %>%
  mutate(x = rescale(temp, c(-0.5, 0.5))) %>%
  arrange(X1) %>%
  group_by(y) %>%
  filter(lead(x) < x & max(cgs) > 40) %>%
  #sample_n(100) %>%
  ungroup()# %>%
  #group_by(y) %>%
  #mutate(rn = row_number()) %>%
  #ungroup()# %>%
  #

df %>% group_by(y) %>%
  mutate(rn = row_number()) %>%
  filter(lead(x) > x) %>%
  ungroup() %>%
  ggplot(aes(temp, cgs)) + geom_point(aes(colour = rn, group = y))

df %>% ggplot(aes(x = temp, y = mag)) +
  geom_point(aes(colour = cgs))

df %>% ggplot(aes(x = temp, y = cgs)) +
  geom_line(aes(color = mag, group = y))

df %>% select(y) %>% unique() %>% sample_n(1)

df2 = df %>% filter(y == -0.1)

df2 %>% ggplot(aes(x = temp, y = cgs)) +
  geom_line()

sdata = list(N = nrow(df2),
             M = 20,
             scale = 0.25,
             x = df2$x,
             y = df2$cgs)

fit = stan("models/approx_gp.stan", data = sdata, chains = 1, iter = 1000)

a = as_tibble(extract(fit, "f")$f)
colnames(a) = 1:ncol(a)
b = as_tibble(list(idx = 1:nrow(df2), temp = df2$temp))
out = inner_join(a %>% gather(idx, f) %>% mutate(idx = as.integer(idx)), b, by = "idx")

summary = out %>% group_by(temp) %>%
  summarize(mean = mean(f),
            q1 = quantile(f, 0.025),
            q2 = quantile(f, 0.167),
            q3 = quantile(f, 0.833),
            q4 = quantile(f, 0.975)) %>%
  ungroup()

summary %>% ggplot(aes(temp, mean)) +
  geom_ribbon(aes(ymin = q1, ymax = q4), alpha = 0.75, fill = "grey") +
  geom_line() +
  geom_line(aes(temp, q1), alpha = 1.0, size = 0.125) +
  geom_line(aes(temp, q4), alpha = 1.0, size = 0.125) +
  geom_point(data = df2, aes(temp, cgs), size = 0.1, col = "red")

#f %>%
fld = interp(x = df$temp, y = df$mag, z = df$cgs, linear = TRUE, jitter = 0.1)

dff = melt(fld$z, na.rm = TRUE)
names(dff) <- c("temp", "mag", "cgs")
dff$temp <- fld$x[dff$temp]
dff$mag <- fld$y[dff$mag]

dff %>% ggplot(aes(x = temp, y = mag, z = cgs)) +
  geom_tile(aes(fill = cgs)) +
  scale_fill_gradientn(colours = terrain.colors(10)) +
  stat_contour()

#ggplot(aes(x = temp, y = mag)) +
#  geom_tile(aes(fill = cgs))
