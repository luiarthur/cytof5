library(nimble)
library(rcommon)
library(cytof3)

# Nimble Model
model.code <- nimbleCode({
  for (n in 1:Nsum) {
    # Y[n, 1:J] ~ dmnorm(mu[n, 1:J], sig2[n] * R[1:J, 1:J])
    for (j in 1:J) {
      Y[n, j] ~ dnorm(mu[n, j], var=sig2[n])
    }
    mu[n, 1:J] <- mu_star[s[n], 1:J]
    sig2[n] <- sig2_star[s[n]]
    for (j in 1:J) {
      logit(p[n, j]) <- b0 + b1 * Y[n, j]
    }
    # mixture component parameters drawn from base measures
    mu_star[n, 1:J] ~ dmnorm(zeroVec[1:J], R[1:J, 1:J])
    sig2_star[n] ~ dinvgamma(.1, .1)
  }
  s[1:Nsum] ~ dCRP(alpha, size=Nsum)
  alpha ~ dgamma(1, 1)
  b0 ~ dnorm(0, 4)
  b1 ~ dnorm(0, 4)
})

# Read and preprocess
y = readRDS('data/cytof_cb.rds')
subsamp_prop = .01
y = lapply(y, function(yi) {
  Ni = NROW(yi)
  yi[sample(1:Ni, as.integer(Ni * subsamp_prop)), ]
})

Y = Reduce(rbind, y)
J = NCOL(Y)

N = sapply(y, NROW)
Nsum = sum(N)
I = length(y)
N_lower = c(0, cumsum(N[-I])) + 1
N_upper = cumsum(N)
idx = lapply(as.list(1:I), function(i) N_lower[i]:N_upper[i])

# Other Model Specs
model.consts = list(Nsum=Nsum, J=J)
model.data = list(Y=Y, R=diag(J), zeroVec=rep(0, J))
model = nimbleModel(model.code, data=model.data, constants=model.consts)

# Compile
cmodel = compileNimble(model, showCompilerOutput=TRUE)

# Configure
model.conf = configureMCMC(model, print=FALSE)
model.conf$addMonitors(c('mu_star', 'sig2_star', 'b0', 'b1', 'alpha', 's'))
print(system.time(
  model.mcmc <- buildMCMC(model.conf) # build time increases as N grows
))
cmodel = compileNimble(model.mcmc, project=model)

# Run
samps = runMCMC(cmodel, summary=TRUE, niter=1000, nburnin=5000)


