# install.packages('rstan')
library(rstan)

# For reproducibility
set.seed(0)

# Simulate data
N = 500
y = rnorm(N, 2, .3)

# Put STAN data in a list
K = 5
data = list(N=N, K=5, y=y, alpha=rep(1, K) / K,
            m_mu=0, s_mu=.1, m_sig=1, s_sig=.5)

# Fit STAN model (NUTS)
fit = stan(file='mix.stan', iter=5000, data=data, seed=0)
