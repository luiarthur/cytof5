# install mclust if necessary
if ("mclust" %in% installed.packages()) {
  library(mclust)
} else {
  cat("Package `mclust` not found. Installing `mclust`...\n")
  install.packages("mclust")
}

subsampleData <- function(y, percentage) {
  if (percentage >= 1 || percentage <= 0) {
    return(y)
  } else {
    N <- NROW(y)
    N_subsample <- as.integer(N * percentage)
    idx_subsample <- sample(1:N, N_subsample, replace=FALSE)
    return(y[idx_subsample, ])
  }
}

preimpute <- function(Y) {
  # impute missing values
  yneg <- Y[Y < 0 & !is.na(Y)]
  yneg <- yneg[yneg < quantile(yneg, .1)]
  num_missing_Y <- sum(is.na(Y))
  # Y[is.na(Y)] <- sample(yneg, size=num_missing_Y, replace=TRUE)
  Y[is.na(Y)] <- rnorm(num_missing_Y, max(yneg), sd=.1)
  stopifnot(sum(is.na(Y)) == 0)

  return(Y)
}

precluster <- function(Y, G, modelNames="VVI") {
  #' Y: list of matrices
  #' G: number of groups in clustering
  # TODO
  Mclust(y_subsample, G=G, modelNames="VVI")
}

get_group <- function(clus, g) {
  clus$data[clus$class == g, ]
}

get_clus_g_stats <- function(clus, g) {
  x = get_group(clus, g)
  mx = colMeans(x)
  list(mean=mx, z=as.integer(mx > 1))
}

get_clus_stats <- function(clus, L_sum) {
  s = lapply(as.list(1:clus$G), function(g) get_clus_g_stats(clus, g=g))
  clus_means = sapply(s, function(sg) sg$mean) # J x G
  Z = sapply(s, function(sg) sg$z) # J x G
  K = ncol(Z)

  # Create mus0
  mus = quantile(clus_means, seq(0, 1, len=L_sum))
  mus0 = mus[which(mus >  0)]
  mus1 = mus[which(mus <= 0)]

  # Init W
  W = rep(NA, K)
  W = sapply(2:K, function(k) mean(clus$class == k))
  W[1] = 1 - sum(W[-1])

  # Init alpha
  alpha = 1.0

  list(clus_means=clus_means, mus0=mus0, mus1=mus1, mus=mus, Z=Z, lam=clus$class, W=W,
       alpha=alpha)
}

# Examples:
# plot(clus, what="uncertainty", dimens=c(5,1))
# plot(clus, what="classification", dimens=c(5,1,32))
# plot(clus, what="BIC", dimens=c(5,1,32))

dec2bin <- function(x, out=NULL) {
  #' convert base 10 number to base 2
  if (x == 0) out else {
    d <- x %/% 2
    r <- x %% 2
    dec2bin(d, c(r, out))
  }
}

bin2dec <- function(x, acc=0) {
  #' convert base 2 number to base 10
  if (length(x) == 1) {
    acc + ifelse(x == 1, 1, 0)
  } else {
    n <- length(x)
    binary2bas10(tail(x, n - 1), acc=(2*x[1]) ^ (n-1) + acc)
  }
}

binstr <- function(x) {
  paste0(x, collapse='')
}

str2vec <- function(x) {
  as.integer(strsplit(x, '')[[1]])
}

genCountMap <- function(clus, Z) {
  d = list()
  for (k in clus$class) {
    z = binstr(Z[, k])
    if (z %in% names(d)) {
      d[z] = d[z][[1]] + 1
    } else {
      d[z] = 1
    }
  }

  return(d)
}


# Tests
# pass = all(sapply(sapply(1:1000, dec2bin), bin2dec) == 1:1000)
# stopifnot(pass)
