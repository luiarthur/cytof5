import scipy
import pystan
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import torch
import torch.distributions as dist

model_name = 'mix'

# Compile stan model, if needed. Otherwise, load model.
if os.path.exists('{}.pickle'.format(model_name)):
    # Load model if it is cached.
    sm = pickle.load(open('{}.pickle'.format(model_name), 'rb'))
else:
    # compile model
    sm = pystan.StanModel(file='{}.stan'.format(model_name))
    # save model for later use.
    with open('{}.pickle'.format(model_name), 'wb') as f:
        pickle.dump(sm, f)


# Set random seed
np.random.seed(0)

# Simulate data
N = 500
mu_true = 2
sig_true = .3
y = np.random.randn(N) * sig_true + mu_true

# Fit STAN model (NUTS)
Kmcmc = 3
data = dict(N=N, K=Kmcmc, y=y, alpha=np.ones(Kmcmc) / Kmcmc,
            m_mu=0, s_mu=1, m_sig=1, s_sig=.5)
fit = sm.sampling(data=data, iter=1000, chains=1, seed=0)

# Extract samples 
samples = fit.extract()

# Plot
plt.figure()
plt.subplot(2, 2, 1)
plt.boxplot(samples['w'])
plt.title('w')

plt.subplot(2, 2, 2)
plt.boxplot(samples['mu'])
plt.title('mu')

plt.subplot(2, 2, 3)
plt.boxplot(samples['sig'])
plt.title('sigma')

plt.subplot(2, 2, 4)
plt.plot(samples['lp__'])
plt.title('log posterior')

plt.tight_layout()
plt.show()

# Function to compute log likelihood
def loglike(mu, sig, w):
    return (dist.Normal(torch.from_numpy(mu), torch.from_numpy(sig[:, None]))
            .log_prob(torch.from_numpy(y[:, None])) +
            torch.from_numpy(w).log()).logsumexp(1)

# Compute log likelihood
ll_post = loglike(samples['mu'], samples['sig'], samples['w']).numpy()
ll_true = (dist.Normal(mu_true, sig_true)
           .log_prob(torch.from_numpy(y)).mean().numpy())

# Plot log likelihood
plt.plot(ll_post,
         label='log likelihood: {}'.format(ll_post.mean().round(4)))
plt.axhline(ll_true,
            label='truth: {}'.format(ll_true.round(4)),
            color='orange', lw=3)
plt.legend()
plt.show()
