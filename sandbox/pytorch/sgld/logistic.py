import torch
import time
import math
import copy
import datetime
import matplotlib.pyplot as plt 
import numpy as np

from sgld import sgld
import sys
sys.path.append('..')

def solve_ab(lr_init, lr_end, gam, N):
    b = N / ((lr_init / lr_end) ** (1 / gam) - 1)
    a = lr_init * (b ** gam)
    return (a, b)

# logpdf of Normal
def loglike(y, x, b):
    p = torch.sigmoid(b[0] + b[1] * x)
    return bern_lpdf(y, p).sum()

def bern_lpdf(x, p):
    return x * torch.log(p) + (1 - x) * torch.log(1 - p)

def normal_lpdf(x, m, s):
    return -torch.log(s) - (x - m) ** 2 / (2 * s * s)

def log_prior(b):
    return normal_lpdf(b, torch.tensor(0.), torch.tensor(5.)).sum()

def fit(y, x, nmcmc=1000, nburn=10, lr_init=1e-2, lr_end=1e-4, gam=.55, seed=1,
        minibatch_size:int=0):
    # Check lr params
    assert(lr_init >= lr_end >= 0 and 0 <= gam <= 1)

    # Set random seed for reproducibility
    torch.manual_seed(seed)

    # Define data type
    dtype = torch.float64

    # Define device-type (cpu / gpu)
    device = torch.device("cpu")

    # Number of obs
    N = len(y)

    def log_post(state):
        b = state[0]
        if 0 < minibatch_size < N:
            idx = torch.randperm(N)[:minibatch_size]
            y_minibatch = y[idx]
            x_minibatch = x[idx]
        else:
            y_minibatch = y
            x_minibatch = x
        
        # log likelihood
        ll = (loglike(y_minibatch, x_minibatch, b) / minibatch_size) * N

        return ll + log_prior(b)

    # Initialize parameters.
    b = torch.randn(2, device=device, dtype=dtype)
    b.requires_grad = True
    state = [b]
    log_post_history = [log_post(state).item()]
    out = []
    eps_history = []

    iters = nmcmc + nburn
    (a_eps, b_eps) = solve_ab(lr_init, lr_end, gam, iters)

    for t in range(iters):
        now = datetime.datetime.now()
        print('{} | iteration: {} / {} | normalized logpost: {}'.format(now, 
              t + 1, iters, log_post_history[-1] / N))

        eps_curr = a_eps * (b_eps + t) ** (-gam)
        sgld(log_post, state=state, log_post_history=log_post_history,
             eps=eps_curr)

        if t >= nburn:
            out.append(copy.deepcopy(state))
            eps_history.append(eps_curr)

        print('b: {}'.format(state[0].tolist()))
        print()

    return {'chain': out, 'logpost_hist': log_post_history, 'eps': eps_history,
            'nmcmc': nmcmc, 'nburn': nburn}
  

if __name__ == '__main__':
    torch.manual_seed(1)

    N = 500
    x = torch.randn(N, dtype=torch.float64)
    b0 = .5
    b1 = 2
    p = torch.tensor(torch.sigmoid(b0 + b1 * x), dtype=torch.float64)
    y = torch.tensor(p > torch.rand(N, dtype=torch.float64), dtype=torch.float64)

    # print data
    with open('data/data.txt', 'w') as file:
        file.write('y, x \n')
        for n in range(N):
            file.write('{}, {}\n'.format(y[n], x[n]))

    out = fit(y, x, lr_init=1e-1, lr_end=1e-4, gam=.55,
              nmcmc=1000, nburn=10000, minibatch_size=200, seed=3)

    ll = out['logpost_hist']
    plt.plot(ll)
    plt.axvline(x=out['nburn'], color='red')
    plt.show()

    def predict(x, b):
        return torch.sigmoid(b[0] + b[1] * x)

    x_new = torch.arange(-6, 6, .1, dtype=torch.float64)
    y_new = torch.stack([predict(x_new, c[0]) for c in out['chain']])
    y_mean = y_new.mean(0)
    y_lower = y_mean - 3 * y_new.std(0)
    y_upper = y_mean + 3 * y_new.std(0)
    plt.plot(x_new.tolist(), y_mean.tolist())
    plt.plot(x_new.tolist(), y_lower.tolist())
    plt.plot(x_new.tolist(), y_upper.tolist())
    plt.show()

    b = torch.stack([c[0] for c in out['chain']])
    print('b mean: {}'.format(b.mean(0).tolist()))
    print('b std:  {}'.format(b.std(0).tolist()))

    eps = torch.tensor(out['eps'], dtype=torch.float64)
    b0_mean = (b[:, 0] * eps).sum() / eps.sum()
    b0_std = torch.sqrt((b[:, 0] - b0_mean) ** 2 * eps).sum() / torch.sqrt(eps.sum())

    print('b0 mean: {}'.format(b0_mean.item()))
    print('b0 std:  {}'.format(b0_std.item()))

