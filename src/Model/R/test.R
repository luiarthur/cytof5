set.seed(0)
source('init_Z.R')

y = readRDS('../../../sims/cb/data/cytof_cb.rds')
y = preimpute(y)
N = NROW(y)
idx_subsample = sample(1:N, 500, replace=FALSE)
y_subsample = y[idx_subsample, ]

# KMEANS EXAMPLE
# ks = seq(5, 40, by=5)
# y_clus = lapply(as.list(ks), function(k) {cat('k=', k, '\n'); kmeans(y, centers=k, iter.max=100)})
# 
# tot.withinss = sapply(y_clus, function(o) o$tot.withinss)
# plot(ks, sqrt(tot.withinss / N), type='o')

# MCLUST
y_clus = Mclust(y_subsample)
plot(y_clus, what="uncertainty", dimens=c(5,1))
plot(y_clus, what="classification", dimens=c(5,1,32))
plot(y_clus, what="BIC", dimens=c(5,1,32))


y_clus = Mclust(y_subsample)
