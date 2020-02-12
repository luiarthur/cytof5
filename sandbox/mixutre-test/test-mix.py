import pystan
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

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
y = np.random.randn(N) * .3 + 2
K = 5

# Fit STAN model (NUTS)
data = dict(N=N, K=K, y=y, alpha=np.ones(K) / K,
            m_mu=0, s_mu=.1, m_sig=1, s_sig=.5)
fit = sm.sampling(data=data, iter=1000, chains=1, seed=0)

# Extract samples 
samples = fit.extract()

# Plot
plt.figure()
plt.subplot(3, 1, 1)
plt.boxplot(samples['w'])
plt.title('w')
plt.subplot(3, 1, 2)
plt.boxplot(samples['mu'])
plt.title('mu')
plt.subplot(3, 1, 3)
plt.boxplot(samples['sig'])
plt.title('sigma')
plt.tight_layout()
plt.show()
