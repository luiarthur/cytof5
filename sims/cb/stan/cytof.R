library(rstan)
library(rcommon)

# compiled_stan_model = "compiled_stan_model.rds"
# stan_model = "model.stan" # full model

compiled_stan_model = "compiled_stan_model2.rds"
stan_model = "model2.stan" # no eps, no missing

if (file.exists(compiled_stan_model)) {
  cat("Reading previously-compiled STAN model...", "\n")
  cmodel <- readRDS(compiled_stan_model)
} else {
  cat("Compiling STAN model...", "\n")
  system.time(cmodel <- stan_model(stan_model))
  saveRDS(cmodel, compiled_stan_model)
}

use_vb = TRUE

y = readRDS('../data/cytof_cb.rds')
subsamp_prop = .01
y = lapply(y, function(yi) {
  Ni = NROW(yi)
  yi[sample(1:Ni, as.integer(Ni * subsamp_prop)), ]
})

badCols = c(2, 4, 6, 9, 11, 21, 31)
Y = Reduce(rbind, y)
Y = Y[, -badCols]
Nsum = NROW(Y)
J = NCOL(Y)
N = sapply(y, NROW)
I = length(y)

m = matrix(0, Nsum, J)
m[is.na(Y)] <- 1
Y[is.na(Y)] <- -6
group = unlist(sapply(1:I, function(i) rep(i, N[i])))
data = list(J=J, I=I, K=10, N=Nsum, group=group, L0=5, L1=3,
            m=m, y=Y)
data$a_W = rep(1/data$K, data$K)
data$a_eta0 = rep(1/data$L0, data$L0)
data$a_eta1 = rep(1/data$L1, data$L1)

init = list(mu0=sort(runif(data$L0, -3, -2)),
            mu1=sort(runif(data$L1, 2, 3)),
            sigma=rep(.1, data$I),
            alpha=1,
            v=rep(1/data$K, data$K))

if (use_vb) {
  cat("Using VI...", '\n')
  out <- vb(cmodel, data=data, init=init)
} else {
  cat("Using HMC...", '\n')
  out <- stan(file='model.stan', data=data, init=list(init),
              iter=2000, warmup=1000, chains=1)
}

samps = extract(out)
plot(samps$lp__, type='l')
# plotPosts(samps$mu0)
# plotPosts(samps$sig)
# plotPosts(samps$theta)

