library(rcommon)
library(cytof3)

y = readRDS('data/cytof_cb.rds')
I = length(y)
Y = Reduce(rbind, y)
Y[is.na(Y)] <- -3
N = sapply(y, NROW)
N_lower = c(0, cumsum(N[-I])) + 1
N_upper = cumsum(N)
idx = lapply(as.list(1:I), function(i) N_lower[i]:N_upper[i])

km = kmeans(Y, centers=20, iter.max=100)
clus_ord = order(table(km$clus))

gp = clus_ord[3]
yg = Y[km$clus== gp, ]
# my.image(Y[order(km$clus==gp), ][1:km$size[gp], ], col=blueToRed(9), zlim=c(-4,4), addL=T)

ci = t(apply(yg, 2, quantile, c(.2, .8)))
plot(km$centers[gp, ], ylim=c(-5, 5), pch=4, lwd=3, col='red', ylab=paste('Group', gp))
add.errbar(ci)
abline(h=0)

