data {
  int<lower=1> N;
  int y[N];
  real x[N];
}

parameters {
  real b0;
  real b1;
}

model {
  for (n in 1:N) {
    y[n] ~ bernoulli_logit(b0 + x[n] * b1);
  }

  b0 ~ normal(0, 3);
  b1 ~ normal(0, 3);
}
