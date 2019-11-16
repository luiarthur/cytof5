data {
  int N;
  int K;
  vector[N] y;
  vector<lower=0>[K] alpha;
  real m_mu;
  real<lower=0> s_mu;
  real m_sig;
  real<lower=0> s_sig;
}

parameters {
  ordered[K] mu;
  simplex[K] w;
  real<lower=0> sig;
}

model {
  real log_probs[K];

  mu ~ normal(m_mu, s_mu);
  sig ~ lognormal(m_sig, s_sig);
  w ~ dirichlet(alpha);

  for (i in 1:N) {
    for (k in 1:K) {
      log_probs[k] = log(w[k]) + normal_lpdf(y[i] | mu[k], sig);
    }
    target += log_sum_exp(log_probs);
  }
}
