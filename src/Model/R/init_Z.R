# install mclust if necessary
if ("mclust" %in% installed.packages()) {
  library(mclust)
} else {
  cat("Package `mclust` not found. Installing `mclust`...\n")
  install.packages("mclust")
}

preimpute <- function(y) {
  # concatenate the y
  Y <- do.call(rbind, y)

  # impute missing values
  yneg <- Y[Y < 0 & !is.na(Y)]
  yneg <- yneg[yneg < quantile(yneg, .1)]
  num_missing_Y <- sum(is.na(Y))
  # Y[is.na(Y)] <- sample(yneg, size=num_missing_Y, replace=TRUE)
  Y[is.na(Y)] <- rnorm(num_missing_Y, max(yneg), sd=.1)
  stopifnot(sum(is.na(Y)) == 0)

  return(Y)
}

precluster <- function(Y, G) {
  #' Y: list of matrices
  #' G: number of groups in clustering
  # TODO
}
