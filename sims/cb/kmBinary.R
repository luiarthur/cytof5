library(rcommon)
library(cytof3)

sigmoid = function(x) 1 / (1 + exp(-x))
prob_expr = function(y, s=5) ifelse(is.na(y), 0, sigmoid(s * y))
plot(yy <- seq(-6, 6, len=100), prob_expr(yy), type='l', ylim=c(0, 1))
reorder_labels = function(labels) {
  ord = order(table(labels), decreasing=TRUE)
  out = labels
  K = length(ord)
  for (k in 1:K) {
    out[labels == ord[k]] = k
  }
  out
}

y = readRDS('data/cytof_cb.rds')
I = length(y)
Y = Reduce(rbind, y)
N = sapply(y, NROW)
Nsum = sum(N)
N_lower = c(0, cumsum(N[-I])) + 1
N_upper = cumsum(N)
idx = lapply(as.list(1:I), function(i) N_lower[i]:N_upper[i])
Z = prob_expr(Y)
K = 10
J = NCOL(Y)

km = kmeans(Z, centers=K, iter.max=100)
clus_ord = order(table(km$clus), decreasing=TRUE)
new_clus = reorder_labels(km$clus)

# Z_inj
layout(M <- rbind(matrix(1, 15, 20), matrix(2, 5, 20)))
my.image(Z[order(new_clus), ], col=greys(5))
# my.image(Y[order(new_clus), ], col=blueToRed(9), zlim=c(-4,4), na.col='black')
abline(h = cumsum(table(new_clus)), lwd=3, col="yellow")

# Z_hat
Z_hat = t(sapply(1:K, function(k) colMeans(Z[new_clus == k, ])))
my.image(Z_hat, col=greys(5))
abline(h=1:K - .5, v=1:J - .5, col='grey')
par(mfrow=c(1,1))

# W_hat
# W_hat = t(sapply(1:I, function(i) table(km$clus[idx[[i]]]) / N[i]))
# plot.ts(t(W_hat), type='p', cex=2, pch=20, xlab="groups")
# W_hat_var = t(sapply(1:I, function(i) W_hat[i, ] * (1 - W_hat[i, ]) / N[i]))
# W_hat_lower = W_hat - 1.96 * sqrt(W_hat_var)
# W_hat_upper = W_hat + 1.96 * sqrt(W_hat_var)
# par(mfrow=c(I, 1))
# for (i in 1:I) {
#   plot(W_hat[i, ], pch=20, cex=2)
#   add.errbar(cbind(W_hat_lower[i,], W_hat_upper[i,]))
# }
# par(mfrow=c(1, 1))

# gp = clus_ord[2]
# yg = Y[km$clus== gp, ]
# my.image(yg, col=blueToRed(9), zlim=c(-4,4), addL=T)
# 
# ci = t(apply(yg, 2, quantile, c(.2, .8)))
# plot(colMeans(yg), ylim=c(-5, 5), pch=4, lwd=3, col='red', ylab=paste('Group', gp))
# add.errbar(ci)
# abline(h=0)

