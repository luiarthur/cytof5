library(rstan)
library(rcommon)

compiled_stan_model = "compiled_stan_model.rds"
stan_model = "model.stan"

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
Y[is.na(Y)] <- -5
group = unlist(sapply(1:I, function(i) rep(i, N[i])))
data = list(J=J, I=I, K=10, N=Nsum, group=group, L0=5, L1=3, m=m, y=Y)


init = list(mu0=sort(runif(data$L0, -3, 0)),
            mu1=sort(runif(data$L1, 0, 3)))
if (use_vb) {
  out <- vb(cmodel, data=data, init=init)
} else {
  out <- stan(file='model.stan', data=data, init=list(init),
             iter=2000, warmup=1000, chains=1)
}

