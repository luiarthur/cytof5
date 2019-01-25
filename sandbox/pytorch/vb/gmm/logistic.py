# References: https://chrisorm.github.io/VI-MC-PYT.html
import torch
import math
import copy
import datetime
import matplotlib.pyplot as plt 
import numpy as np
import VB

# Define device-type (cpu / gpu)
device = torch.device("cpu")

# Define data type
dtype = torch.float64

class model(VB.VB):
    def init_v(self):
        b = torch.randn(2, device=self.device, dtype=self.dtype)
        b.requires_grad = True
        return b

    def subsample_data(self, data, minibatch_info=None):
        if minibatch_info is None:
            mini_data = data
        else:
            n = mini_data['size']
            N = data['x'].size().numel()
            idx = np.random.choice(N, n, replace=False)
            mini_data = {'x': data['x'][idx], 'y': data['y'][idx]}
        return mini_data

    def sample_real_params(self, v):
    def log_q(self, real_params, v):
    def log_prior(self, real_params):
    def loglike(self, params, data, minibatch_info=None):

    def to_real_space(self, params):
    def to_param_space(self, real_params):

if __name__ == '__main__':
    N = 500
    x = np.random.randn(N)
    b0 = .5
    b1 = 2.
    p = 1 / (1 + np.exp(-(b0 + b1 * x)))
    y = (p > np.random.rand(N)) * 1.0

    x = torch.tensor(x)
    y = torch.tensor(y)
    data = {'x': x, 'y': y}
    out = model.fit(data, lr=1e-2, minibatch_info={'size': 500},
                    niters=5000, nmc=10, seed=2, eps=1e-6, init=None,
                    print_freq=50)

    # ELBO
    elbo = np.array(out['elbo'])
    plt.plot(elbo); plt.show()
    plt.plot(np.abs(elbo[101:] / elbo[100:-1] - 1)); plt.show()

    # Posterior Distributions
    b_vp = [s.tolist() for s in out['v']]
    print('b0 mu: {}, sd: {}'.format(b_vp[0][0], np.exp(b_vp[0][1])))
    print('b1 mu: {}, sd: {}'.format(b_vp[1][0], np.exp(b_vp[1][1])))

    # R:
    # Coefficients:
    # Estimate Std. Error z value Pr(>|z|)
    # (Intercept)   0.5010     0.1214   4.126 3.69e-05 ***
    #
