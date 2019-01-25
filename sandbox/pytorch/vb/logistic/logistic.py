# References: https://chrisorm.github.io/VI-MC-PYT.html
import torch
import time
import math
import copy
import datetime
import matplotlib.pyplot as plt 
import numpy as np

import sys
sys.path.append('..')

# Define data type
dtype = torch.float64

# Define device-type (cpu / gpu)
device = torch.device("cpu")

def loglike(y, x, b):
    p = torch.sigmoid(b[0] + b[1] * x)
    return torch.distributions.Bernoulli(p).log_prob(y).sum()

def log_prior(b):
    return torch.distributions.Normal(0., 5.).log_prob(b).sum()

def log_q(b, v):
    """
    b: parameters in logistic regression
    v: variational parameters, in real scale
    """
    return sum([torch.distributions.Normal(v[j][0],
                torch.exp(v[j][1])).log_prob(b[j]) for j in range(len(b))])

def fit(y, x, niters=1000, nsamps=10, lr=1e-3, minibatch_size:int=0, seed=1,
        eps=1e-8, init=None):
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Number of obs
    N = len(y)
    if minibatch_size <= 0 or minibatch_size > N:
        minibatch_size = N

    def compute_elbo(state):
        if 0 < minibatch_size < N:
            # idx = torch.randperm(N)[:minibatch_size]
            idx = np.random.choice(N, minibatch_size, replace=False)
            y_minibatch = y[idx]
            x_minibatch = x[idx]
        else:
            y_minibatch = y
            x_minibatch = x
        
        def compute_one_elbo():
            J = len(state)
            eta = [torch.distributions.Normal(0, 1).sample() for j in range(J)]
            b = torch.stack([eta[j] * state[j][1] + state[j][0] for j in range(J)])
            ll = N * loglike(y_minibatch, x_minibatch, b) / minibatch_size
            out = ll + log_prior(b) - log_q(b, state)
            return out

        return torch.stack([compute_one_elbo() for i in range(nsamps)]).mean()

    # Initialize parameters.
    def create_vp():
        b = torch.randn(2, device=device, dtype=dtype)
        b.requires_grad = True
        return b

    if init is not None:
        state = copy.deepcopy(init)
    else:
        state = [create_vp() for i in range(2)]

    optimizer = torch.optim.Adam(state, lr=lr)
    elbo_history = []
    lr_history = [lr]

    for t in range(niters):
        elbo_mean = compute_elbo(state)
        loss = -elbo_mean
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        elbo_history.append(elbo_mean.item())
        now = datetime.datetime.now()
        print('{} | iteration: {} / {} | elbo: {}'.format(now,
              t + 1, niters, elbo_history[-1]))
        print('state: {}'.format([s.tolist() for s in state]))

        if t > 0 and abs(elbo_history[-1] / elbo_history[-2] - 1) < eps:
            break

    return {'state': state, 'lr': lr_history, 'elbo': elbo_history}


if __name__ == '__main__':
    torch.manual_seed(1)
    np.random.seed(1)

    N = 500
    x = np.random.randn(N)
    b0 = .5
    b1 = 2
    p = 1 / (1 + np.exp(-(b0 + b1 * x)))
    y = (p > np.random.rand(N)) * 1.0

    x = torch.tensor(x)
    y = torch.tensor(y)

    out = fit(y, x, lr=1e-1, minibatch_size=500, niters=1000,
              nsamps=10, seed=1, eps=1e-8, init=None)
    plt.plot(out['elbo']); plt.show()

    # posterior samples
    B = 1000
    b_vp = [s.tolist() for s in out['state']]

    print('b0 mu: {}, sd: {}'.format(b_vp[0][0], np.exp(b_vp[0][1])))
    print('b1 mu: {}, sd: {}'.format(b_vp[1][0], np.exp(b_vp[1][1])))

    # R:
    # Coefficients:
    #             Estimate  Std. Error  z value  Pr(>|z|)
    # (Intercept)   0.5289      0.1226    4.314   1.6e-05 ***
    # x             2.0999      0.1910   10.992   < 2e-16 ***


