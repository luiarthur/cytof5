data {
  int J;
  int I;
  int K;
  int N;
  int<lower=1> group[N]; 
  real y[N, J];
  int<lower=0, upper=1> m[N, J];
  vector<lower=0>[K] a_W;
}

parameters {
  simplex[K] W[I];
  matrix<lower=-10, upper=0>[J, K] mu0;
  matrix<lower=0, upper=10>[J, K] mu1;
  vector<lower=0>[I] sigma;
  real<lower=0> alpha;
  vector<lower=0, upper=1>[K] v;
}

model {
  for (i in 1:I) {
    W[i] ~ dirichlet(a_W);
  }

  sigma ~ gamma(1, 1);
  for (j in 1:J) for (k in 1:K) {
    mu0[j, k] ~ uniform(-10, 0);
    mu1[j, k] ~ uniform(0, 10);
  }

  alpha ~ gamma(1, 1);
  v ~ beta(alpha / K, 1);

  for (n in 1:N) {
    int i = group[n];

    real ll[K];
    for (k in 1:K) {
      ll[k] = log(W[i][k]);
      for (j in 1:J) {
        ll[k] += m[n, j] * log_sum_exp(log(1 - v[k]) + normal_lcdf(y[n, j] | mu0[j, k], sigma[i]),
                                       log(v[k]) + normal_lcdf(y[n, j] | mu1[j, k], sigma[i])) +
                 (1 - m[n, j]) * log_sum_exp(log(1 - v[k]) + normal_lpdf(y[n, j] | mu0[j, k], sigma[i]),
                                             log(v[k]) + normal_lpdf(y[n, j] | mu1[j, k], sigma[i]));
      }
    }
    target += log_sum_exp(ll);
  }
}


