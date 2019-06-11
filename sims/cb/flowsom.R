library(rcommon)
library(cytof3)

y = readRDS('data/cytof_cb.rds')
I = length(y)
Y = Reduce(rbind, y)
Y[is.na(Y)] <- -6
N = sapply(y, NROW)
N_lower = c(0, cumsum(N[-I])) + 1
N_upper = cumsum(N)
idx = lapply(as.list(1:I), function(i) N_lower[i]:N_upper[i])
J = NCOL(Y)

colnames(Y) <- 1:NCOL(Y)
ff_Y = flowCore::flowFrame(Y)

Ks = seq(3, 33, by=3)
models = lapply(as.list(Ks),
                function(K) {print(K);
                  FlowSOM::FlowSOM(ff_Y, colsToUse=1:J, nClus=K, seed=42)})

get.clus = function(mod) {
  mod$meta[mod$FlowSOM$map$mapping[,1]]
}

get.variance = function(mod) {
  clus = get.clus(mod)
  uclus = unique(clus)
  vars = sapply(uclus, function(k) sum(apply(Y[clus == k, ], 2, var)))
  mean(vars)
}

THRESH = .01
num.small.clus = sapply(models, function(mod) {
  clus = get.clus(mod)
  tab = table(clus)
  sum(tab / sum(N) < THRESH)
})
vars = sapply(models, get.variance)
plot(Ks, vars, type='o')

plot(num.small.clus, vars, type='l')
text(num.small.clus, vars, Ks)
