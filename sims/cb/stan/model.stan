data {
  int J;
  int I;
  int K;
  int N;
  int group[N]; 
  int L0;
  int L1;
  int<lower=0, upper=1> m[N, J];
  real y[N, J];
}

parameters {
  simplex[K] W[I];
  ordered[L0] mu0;
  ordered[L1] mu1;
  vector<lower=0>[I] sigma;

  simplex[L0] eta0[I, J];
  simplex[L1] eta1[I, J];

  real<lower=0> alpha;
  vector<lower=0, upper=1>[K] v;

  real<lower=0, upper=1> eps[I];
}

model {
  for (i in 1:I) {
    W[i] ~ dirichlet(rep_vector(1.0 / K, K));
    sigma[i] ~ gamma(1, 1);
  }

  for (l in 1:L0) mu0[l] ~ normal(-3, 1)T[-10, 0];
  for (l in 1:L1) mu1[l] ~ normal(3, 1)T[0, 10];

  for (i in 1:I) {
    for (j in 1:J) {
      eta0[i, j] ~ dirichlet(rep_vector(1.0 / L0, L0));
      eta1[i, j] ~ dirichlet(rep_vector(1.0 / L1, L1));
    }
    eps[i] ~ beta(5, 95);
  }

  alpha ~ gamma(1, 1);

  for (k in 1:K) {
    v[k] ~ beta(alpha / K, 1);
  }

  for (n in 1:N) {
    int i = group[n];
    real A = eps[i];
    real B = 0;

    for (j in 1:J) {
      if (m[n, j] == 1) { // if missing
        A *= normal_cdf(y[n, j], 0, 3);
      } else {
        A *= exp(normal_lpdf(y[n, j] | 0, 3));
      }
    }

    for (k in 1:K) {
      real x = 0;
      for (j in 1:J) {
        real x0 = 0;
        real x1 = 0;

        for (l in 1:L0) {
          if (m[n, j] == 1) { // if missing
            // x0 += eta0[i, j][l] * normal_cdf(y[n, j], mu0[l], sigma[i]);
            x0 += eta0[i, j][l] * normal_cdf(y[n, j], -5, sigma[i]);
          } else {
            x0 += eta0[i, j][l] * exp(normal_lpdf(y[n, j] | mu0[l], sigma[i]));
          }
        }
        x0 *= v[k];

        for (l in 1:L1) {
          if (m[n, j] == 1) { // if missing
            // x1 += eta1[i, j][l] * normal_cdf(y[n, j], mu1[l], sigma[i]);
            x1 += eta1[i, j][l] * normal_cdf(y[n, j], -5, sigma[i]);
          } else {
            x1 += eta1[i, j][l] * exp(normal_lpdf(y[n, j] | mu1[l], sigma[i]));
          }
        }
        x1 *= (1 - v[k]);

        B += (x0 + x1) * W[i][k];
      }
    }
    B *= (1 - eps[i]);

    target += log(A + B);
  }
}


