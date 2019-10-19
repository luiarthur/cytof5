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
  vector<lower=0>[L0] delta0;
  vector<lower=0>[L1] delta1;
  vector<lower=0>[I] sigma;

  simplex[L0] eta0[I, J];
  simplex[L1] eta1[I, J];

  real<lower=0> alpha;
  vector<lower=0, upper=1>[K] v;
  matrix[J, K] H;
}

transformed parameters {
  vector[L0] mu0; 
  vector[L1] mu1; 
  vector<lower=0, upper=1>[K] v_cumprod;
  int Z[J, K];

  mu0 <- -cumsum(delta0);
  mu1 <- cumsum(delta1);
  v_cumprod <- cumprod(v);
  for (j in 1:J) for (k in 1:K) {
    Z[j, k] <- v_cumprod > std_normal_cdf(H[j, k]) ? 1 : 0;
  }
}

model {
  for (i in 1:I) {
    W[i] ~ dirichlet(rep_vector(1.0 / K, K));
    sigma[i] ~ gamma(1, 1);
  }

  for (l in 1:L0) delta0[l] ~ gamma(1, 1);
  for (l in 1:L1) delta1[l] ~ gamma(1, 1);

  for (i in 1:I) {
    for (j in 1:J) {
      eta0[i, j] ~ dirichlet(rep_vector(1.0 / L0, L0));
      eta1[i, j] ~ dirichlet(rep_vector(1.0 / L1, L1));
    }
  }

  alpha ~ gamma(1, 1);

  for (k in 1:K) {
    v[k] ~ beta(alpha, 1);
  }

  // target:
}


