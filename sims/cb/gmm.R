library(rcommon)
library(nimble)
library(cytof3)
set.seed(1)

# Nimble Model
model.code <- nimbleCode({
  # Missing mechanism parameters
  b0 ~ dnorm(0, sd=5)
  b1 ~ dgamma(1, rate=1)

  for (i in 1:I) {
    W[i, 1:K] ~ ddirch(a_W[1:K])
  }

  for (k in 1:K) {
    # mixture component parameters drawn from base measures
    # for (j in 1:J) { mu[j, k] ~ dnorm(0, 1) }
    mu[1:J, k] ~ dmnorm(zeroVec[1:J], R[1:J, 1:J])
    sig2[k] ~ dinvgamma(.1, .1)
  }

  for (n in 1:Nsum) {
    lam[n] ~ dcat(W[idxGroup[n], 1:K])
    for (j in 1:J) {
      # Sampling Density
      Y[n, j] ~ dnorm(mu[j, lam[n]], var=sig2[lam[n]])
      # Missing mechanism
      Q[n, j] ~ dbern(p[n, j])
      logit(p[n, j]) <- b0 - b1 * Y[n, j]
    }
  }
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
idxGroup = unlist(sapply(1:I, function(i) rep(i, N[i])))
Q = is.na(Y) * 1
K = 5
Y.init = Y
Y.init[ which(is.na(Y))] <- -3
Y.init[-which(is.na(Y))] <- NA
W.init = matrix(1/K, I, K)

# Other Model Specs
print("Filling Model Specs ...")
model.data = list(Y=Y, Q=Q)
model.consts = list(Nsum=Nsum, J=J, idxGroup=idxGroup, K=K,
                    a_W=rep(1/K, K), I=I,
                    zeroVec=rep(0, J), R=diag(J))
model.inits = list(Y=Y.init, lam=sample(1:K, Nsum, replace=TRUE),
                   mu=matrix(rnorm(J*K), J, K), sig2=rep(1, K),
                   b0=-2, b1=2, W=matrix(1/K, I, K),
                   p=matrix(.5, Nsum, J))
model = nimbleModel(model.code,
                    data=model.data,
                    constants=model.consts,
                    inits=model.inits)
model$simulate()
model$initializeInfo()

# Compile
print("Compile Stage I ...")
cmodel = compileNimble(model, showCompilerOutput=TRUE)

# Configure
print("Configure ...")
model.conf = configureMCMC(model, print=FALSE)
model.conf$addMonitors(c('mu', 'sig2', 'b0', 'b1', 'lam', 'W'))
print(system.time(
  model.mcmc <- buildMCMC(model.conf) # build time increases as N grows
))
print("Compile Stage II ...")
cmodel = compileNimble(model.mcmc, project=model)

# Run
print("Run Model ...")
nburnin = 10000
niter = nburnin + 1000
samps = runMCMC(cmodel, summary=TRUE, niter=niter, nburnin=nburnin)
# samps = runMCMC(cmodel, summary=TRUE, niter=500+100, nburnin=500)


### Summary ###
get_param = function(name, out_samples) {
  which(sapply(colnames(out_samples), function(cn) grepl(name, cn)))
}

# lam.cols = get_param('lam', samps$samples)
# lam_post= samps$samples[, lam.cols]
# 
# sig2.cols = get_param('sig2', samps$samples)
# sig2_post = samps$samples[, sig2.cols]
# plotPosts(sig2_post)
# 
# mu.cols = get_param('mu', samps$samples)
# mu_post = samps$samples[, mu.cols[1]]
# plotPost(mu_post)
# 
# b0_post = samps$samples[, 'b0']
# b1_post = samps$samples[, 'b1']
# plotPosts(cbind(b0_post, b1_post))
# 
# # Plot posterior density
# 
