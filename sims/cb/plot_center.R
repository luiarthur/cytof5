library(rcommon)
library(cytof3)
system('mkdir -p results/thresh')

y = readRDS('data/cytof_cb.rds')
I = length(y)
Y = Reduce(rbind, y)
J = NCOL(Y)

THRESH = seq(.1, 1, by=.1)
for (thresh in THRESH) {
  M = matrix(NA, I, J)
  for (i in 1:I) for (j in 1:J) {
    M[i, j] = mean(abs(y[[i]][, j]) < thresh, na.rm=TRUE)
  }

  pdf('results/thresh/thresh_' %+% thresh %+% '_.pdf')
  par(mfrow=c(I, 1))
  for (i in 1:I) {
    plot(M[i, ], type='h', main='proportion of cells s.t. |y_{inj}| < ' %+% thresh,
         xlab='markers in sample ' %+% thresh, ylab='proportion')
  }
  par(mfrow=c(1, 1))
  dev.off()
}
