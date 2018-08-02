library(tidyverse)
library(rstan)
library(shinystan)
library(loo)
library("bayesplot")

rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

data = read_csv("SG_steering_data.csv")
stan_data = list(N = nrow(data), near_dot = data$near_dot, far_dot = data$far_dot, near = data$near, dt = 1/30.0, heading_dot = data$heading_dot)

#Fit model
fit1 <- stan(
  file = "SG_steering.stan",  # Stan program
  data = stan_data,    # named list of data
  chains = 4,             # number of Markov chains
  warmup = 1000,          # number of warmup iterations per chain
  iter = 2000,            # total number of iterations per chain
  cores = 4,            # number of cores (using 2 just for the vignette)
  control = list(adapt_delta = 0.8)
)

# launch_shinystan(fit1)
log_lik_1 = extract_log_lik(fit1)
s1 = loo(log_lik_1)
