data {
  int N;
  int<lower=2> K;
  vector[N] y;
  real<lower=0> alpha;
  real m_mu;
  real<lower=0> s_mu;
  real m_sig;
  real<lower=0> s_sig;
}

parameters {
  vector[K] mu;
  vector<lower=0, upper=1>[K] v;
  real<lower=0> sig;
}

transformed parameters {
  simplex[K] w;

  w[1] = v[1];
  for (k in 2:(K - 1)) {
    w[k] = v[k] * (1 - v[k - 1]) * w[k - 1] / v[k - 1];
  }
  w[K] = 1 - sum(w[1:(K-1)]);
}

model {
  real log_probs[K];

  mu ~ normal(m_mu, s_mu);
  sig ~ lognormal(m_sig, s_sig);
  v ~ beta(1, alpha);

  for (i in 1:N) {
    for (k in 1:K) {
      log_probs[k] = log(w[k]) + normal_lpdf(y[i] | mu[k], sig);
    }
    target += log_sum_exp(log_probs);
  }
}
