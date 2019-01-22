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
  vector[K] a_W;
  vector[L0] a_eta0;
  vector[L1] a_eta1;
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
}

model {
  for (i in 1:I) {
    W[i] ~ dirichlet(a_W);
  }

  sigma ~ gamma(1, 1);
  mu0 ~ uniform(-10, 0);
  mu1 ~ uniform(0, 10);

  for (i in 1:I) {
    for (j in 1:J) {
      eta0[i, j] ~ dirichlet(a_eta0);
      eta1[i, j] ~ dirichlet(a_eta1);
    }
  }

  alpha ~ gamma(1, 1);
  v ~ beta(alpha / K, 1);

  for (n in 1:N) {
    int i = group[n];

    real ll[K];
    for (k in 1:K) {
      real logdmix = 0; // prod_{j=1}^J {(1-v_k) * dmix0 + v_k * dmix1}
      real logdmix0 = 0;
      real logdmix1 = 0;

      for (j in 1:J) {
        real lden_mu0[L0];
        real lden_mu1[L1];

        for (l in 1:L0) {
          lden_mu0[l] = normal_lpdf(y[n, j] | mu0[l], sigma[i]);
        }
        logdmix0 = log_mix(eta0[i, j], lden_mu0);

        for (l in 1:L1) {
          lden_mu1[l] = normal_lpdf(y[n, j] | mu1[l], sigma[i]);
        }
        logdmix1 = log_mix(eta1[i, j], lden_mu1);

        logdmix += log_sum_exp(log(1 - v[k]) + logdmix0,
                               log(v[k]) + logdmix1);
      }

      ll[k] = logdmix + log(W[i][k]);
    }

    target += log_sum_exp(ll);
  }
}


