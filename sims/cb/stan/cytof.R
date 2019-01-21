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
badCols = c(2, 4, 6, 9, 11, 21, 31)
Y = Reduce(rbind, y)
Y = Y[, -badCols]
Nsum = NROW(Y)
J = NCOL(Y)
N = sapply(y, NROW)
I = length(y)

m = matrix(0, N, J)
Y[is.na(Y)] <- -3
group = unlist(sapply(1:I, function(i) rep(i, N[i])))
data = list(J=J, I=I, K=10, N=Nsum, group=group, L0=5, L1=3, m=m, y=Y)


out <- vb(cmodel, data=data)
