library(cytof3)
library(rcommon)
library(nimble)
set.seed(1)

# Nimble Model
model.code <- nimbleCode({
  # Missing mechanism parameters
  b0 ~ dunif(-50, -10)
  b1 ~ dunif(-50, -10)

  for (i in 1:I) {
    W[i, 1:K] ~ ddirch(a_W[1:K])
  }

  for (k in 1:K) {
    # mixture component parameters drawn from base measures
    for (j in 1:J) {
      mu[j, k] ~ dnorm(0, 1)
      sig2[j, k] ~ dinvgamma(3, 2)
    }
  }

  for (n in 1:Nsum) {
    lam[n] ~ dcat(W[idxGroup[n], 1:K])
    for (j in 1:J) {
      # Sampling Density
      Y[n, j] ~ dnorm(mu[j, lam[n]], var=sig2[j, lam[n]])
      # Missing mechanism
      Q[n, j] ~ dbern(p[n, j])
      logit(p[n, j]) <- b0 + b1 * Y[n, j]
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
K = 10
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
                   mu=matrix(rnorm(J*K), J, K), sig2=matrix(1, J, K),
                   b0=-2, b1=-2, W=matrix(1/K, I, K),
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
system.time(samps <- runMCMC(cmodel, summary=TRUE, niter=niter, nburnin=nburnin))


### Summary ###
RESULTS_DIR = 'results/gmm/'
'%+%' <- function(a, b) paste0(a, b)

saveRDS(samps, RESULTS_DIR %+% 'mcmc.rds')
# samps = readRDS(RESULTS_DIR %+% 'mcmc.rds')

get_param = function(name, out_samples) {
  which(sapply(colnames(out_samples), function(cn) grepl(name, cn)))
}

lam.cols = get_param('lam', samps$samples)
lam_post= samps$samples[, lam.cols]

sig2.cols = get_param('sig2', samps$samples)
sig2_post = samps$samples[, sig2.cols]
pdf(RESULTS_DIR %+% 'sig2.pdf')
plotPosts(sig2_post[, 1:5])
dev.off()

mu.cols = get_param('mu', samps$samples)
mu_post = samps$samples[, mu.cols]
pdf(RESULTS_DIR %+% 'mu.pdf')
plotPosts(mu_post[, 1:5])
dev.off()

pdf(RESULTS_DIR %+% 'W.pdf')
par(mfrow=c(I, 1))
for (i in 1:I) {
  W.cols = get_param('W\\[' %+% i %+% '\\, ', samps$samples)
  Wi_post = samps$samples[, W.cols]
  boxplot(Wi_post, col='steelblue', pch=20, cex=.5, main='i: ' %+% i)
}
par(mfrow=c(1, 1))
dev.off()

b0_post = samps$samples[, 'b0']
b1_post = samps$samples[, 'b1']
pdf(RESULTS_DIR %+% 'beta.pdf')
plotPosts(cbind(b0_post, b1_post))
dev.off()

## grid of y values
YGRID = seq(-6, 6, l=30)
sigmoid = function(x) 1 / (1 + exp(-x))

pdf(RESULTS_DIR %+% 'prob_miss.pdf')
prob_miss = function(state) {
  sigmoid(state['b0'] + state['b1'] * YGRID)
}
pm = apply(samps$samples, 1, prob_miss)
pm_mean = rowMeans(pm)
plot(YGRID, pm_mean, type='l')
pm_ci = apply(pm, 1, quantile, c(.025, .975))
color.btwn(YGRID, pm_ci[1, ], pm_ci[2, ], from=min(YGRID), to=max(YGRID), 
           col=rgba('blue', .1))
dev.off()


# Plot posterior density
dden_obs = function(state, i, j, ygrid=YGRID) {
  lami = state[lam.cols][idxGroup == i]
  mij = Q[idxGroup == i, j]
  lami = lami[mij == 0]
  sig2 = state[sig2.cols]
  mu = matrix(state[mu.cols], J, K)
  b0 = state['b0']
  b1 = state['b1']
  
  sapply(ygrid, function(yg) mean(dnorm(yg, mu[j, lami], sqrt(sig2[lami]))))
}

for (i in 1:I) for (j in 1:J) {
  print('i:' %+% i %+% ' | j: ' %+% j)
  pdf(RESULTS_DIR %+% 'dden_i' %+% i %+% '_j' %+% j %+% '.pdf')
  dden_ij = apply(samps$samples, 1, dden_obs, i, j)
  dden_mean = rowMeans(dden_ij)
  dden_ci = apply(dden_ij, 1, quantile, c(.025, .975))
  plot(density(y[[i]][, j], na.rm=TRUE), type='l', xlim=range(YGRID), main='')
  lines(YGRID, dden_mean, lwd=3, col='steelblue')
  color.btwn(YGRID, dden_ci[1, ], dden_ci[2, ], from=min(YGRID), to=max(YGRID), 
             col=rgba('blue', .1))
  dev.off()
}
