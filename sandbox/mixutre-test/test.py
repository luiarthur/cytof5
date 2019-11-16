import pystan
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from pystan_vb_extract import pystan_vb_extract

###  Model Name ###
model_name = 'mix'
# model_name = 'mix-dp'

### Use variational inference? ###
# use_vb = True
use_vb = False

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
y = np.random.randn(N) * .3 + 2
# y = np.random.randn(N) * .3  # doesn't work as well
K = 5

# Fit STAN model (NUTS)
data = dict(N=N, K=K, y=y, alpha=np.ones(K) / (10 * K),
            m_mu=0, s_mu=.1, m_sig=1, s_sig=.5)

if model_name == 'mix-dp':
    data['alpha'] = .5

if use_vb:
    fit = sm.vb(data=data, iter=10000, seed=1)
else:
    fit = sm.sampling(data=data, iter=1000, chains=1, seed=0)

# Extract samples 
if use_vb:
    samples = pystan_vb_extract(fit)
else:
    samples = fit.extract()

# store params
mu = samples['mu']
sig = samples['sig']
w = samples['w']

print('mu: {}'.format(mu.mean(0)))
print('w: {}'.format(w.mean(0)))
print('sig: {}'.format(sig.mean()))

