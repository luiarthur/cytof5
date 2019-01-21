library(rstan)
library(rcommon)

compiled_stan_model = "compiled_stan_model.rds"
stan_model = "model.stan"

if (file.exists(compiled_stan_model)) {
  cat("Reading previously-compiled STAN model...", "\n")
  m <- readRDS(compiled_stan_model)
} else {
  cat("Compiling STAN model...", "\n")
  system.time(m <- stan_model(stan_model))
  saveRDS(m, compiled_stan_model)
}

use_vb = TRUE

set.seed(1)
b0 = .5
b1 = 2
sigmoid = function(x) 1 / (1 + exp(-x))
N = 100
x = rnorm(N)
y = rbinom(N, size=1, prob=sigmoid(b0 + b1 * x))

data = list(y=y, x=x, N=N)

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
plotPosts(cbind(samps$b0, samps$b1))

truth = c(b0, b1, NA)
summary_out = summary(out)$summary
if (use_vb) {
  print(cbind(summary_out[, 1], truth, summary_out[, c(3, 7)]))
} else {
  print(cbind(summary_out[, 1], truth, summary_out[, c(4, 8)]))
}

x_grid = seq(-5, 5, len=100)
pred = sapply(x_grid, function(xi) sigmoid(samps$b0 + samps$b1 * xi))
ci = apply(pred, 2, quantile, c(.025, .975))
plot(x_grid, colMeans(pred), type='o', pch=20)
color.btwn(x_grid, ci[1,], ci[2,], from=-10, to=10, col=rgba('blue', .3))
