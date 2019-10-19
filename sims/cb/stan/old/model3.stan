data {
  int J;
  int I;
  int K;
  int N;
  int group[N]; 
  real y[N, J];
  vector[K] a_W;
}

parameters {
  simplex[K] W[I];
  matrix[J, K] mu0;
  matrix[J, K] mu1;
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
    mu0[j, k] ~ normal(-3, 1)T[-10, 0];
    mu1[j, k] ~ normal(3, 1)T[0, 10];
  }

  alpha ~ gamma(1, 1);
  v ~ beta(alpha / K, 1);

  for (n in 1:N) {
    int i = group[n];

    real ll[K];
    for (k in 1:K) {
      real logdmix = 0;
      for (j in 1:J) {
        logdmix += log_sum_exp(log(1 - v[k]) + normal_lpdf(y[n, j] | mu0[j, k], sigma[i]),
                               log(v[k])     + normal_lpdf(y[n, j] | mu1[j, k], sigma[i]));
      }
      ll[k] = logdmix + log(W[i][k]);
    }

    target += log_sum_exp(ll);
  }
}


