library(rstan)
library(rcommon)

compile_stan_model = "compile_stan_model.rds"
stan_model = "model.stan"

if (file.exists(compile_stan_model)) {
  m <- readRDS(compile_stan_model)
} else {
  system.time(m <- stan_model(stan_model))
  saveRDS(m, "")
}

use_vb = TRUE

set.seed(1)
mu = c(-2, 1, 3)
sig = c(.1, .2, .1)
theta = c(.5, .2, .3)
K = length(mu)
N = 100
lam = sample(1:K, N, prob=theta, replace=TRUE)
y = rnorm(N, mu[lam], sig[lam])

data = list(y = y, N=N, K=K)

if (use_vb) {
  # Variational inference
  out <- vb(m, data=data)
} else {
  # HMC
  nburn = 1000
  nmcmc = 500
  out = stan(file='model.stan', data=data,
                 iter=nburn + nmcmc, warmup=nburn, chains=1)
}


# Plot results
samps = extract(out)
plot(samps$lp__, type='l')
plotPosts(samps$mu)
plotPosts(samps$sig)
plotPosts(samps$theta)

truth = c(theta, mu, sig, NA)
summary_out = summary(out)$summary
cbind(summary_out[, 1], truth, summary_out[, c(3,7)])
