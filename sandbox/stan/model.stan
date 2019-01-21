data {
  int<lower=1> K;          // number of mixture components
  int<lower=1> N;          // number of data points
  real y[N];               // observations
}

parameters {
  simplex[K] theta;          // mixing proportions
  ordered[K] mu;             // locations of mixture components
  vector<lower=0>[K] sigma;  // scales of mixture components
}

model {
  // cache log calculation
  vector[K] log_theta = log(theta);

  sigma ~ gamma(1, 1);
  mu ~ normal(0, 3);

  for (n in 1:N) {
    vector[K] lfc_components = log_theta;

    for (k in 1:K) {
      lfc_components[k] += normal_lpdf(y[n] | mu[k], sigma[k]);
    }

    target += log_sum_exp(lfc_components);
  }
}
