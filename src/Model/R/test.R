set.seed(0)
source('init_Z.R')

y = readRDS('../../../sims/cb/data/cytof_cb.rds')
y = lapply(y, preimpute)
y = lapply(y, function(yi) subsampleData(yi, .01))

# MCLUST
clus = Mclust(y[[1]], modelNames="VVI", G=5)

source('init_Z.R')
L_sum = 10
s = get_clus_stats(clus, L_sum)
# d = genCountMap(clus, Z)

plot(s$mus); abline(h=0)
cytof3::my.image(t(s$Z))

# Init Z
# Init mu*
