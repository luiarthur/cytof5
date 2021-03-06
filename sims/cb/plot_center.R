library(rcommon)
library(cytof3)
system('mkdir -p results/thresh')
system('mkdir -p results/thresh/old')

y = readRDS('data/cytof_cb.rds')
I = length(y)
Y = Reduce(rbind, y)
J = NCOL(Y)

good_markers = c(TRUE, FALSE, TRUE, FALSE, TRUE, FALSE, TRUE, TRUE, FALSE,
                 TRUE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE,
                 TRUE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE,
                 TRUE, FALSE, TRUE)

THRESH = seq(.1, 1, by=.1)
for (thresh in THRESH) {
  pdf('results/thresh/thresh_' %+% thresh %+% '_props_.pdf')
  par(mfrow=c(I, 1))
  M = sapply(as.list(1:I), function(i) {
    x = apply(y[[i]][, good_markers], 1, function(yin) sum(abs(yin) < thresh, na.rm=TRUE))
    plot(table(x) / NROW(y[[i]]), ylab='proportions',
         xlab='number of markers (j)',
         main='counts of near-zero expressions in j markers for sample ' %+% i)
    x
  })
  par(mfrow=c(1, 1))
  dev.off()
}

for (thresh in THRESH) {
  gm_idx = which(good_markers)
  M = matrix(NA, I, length(gm_idx))
  for (i in 1:I) for (j in 1:length(gm_idx)) {
    M[i, j] = sum(abs(y[[i]][, gm_idx[j]]) < thresh, na.rm=TRUE) / NROW(y[[i]])
  }

  pdf('results/thresh/old/thresh_' %+% thresh %+% '_.pdf')
  par(mfrow=c(I, 1))
  for (i in 1:I) {
    plot(M[i, ], type='h', main='proportion of cells s.t. |y_{inj}| < ' %+% thresh,
         xlab='markers in sample ' %+% i, ylab='proportion')
    abline(h=.25)
  }
  par(mfrow=c(1, 1))
  dev.off()
}
