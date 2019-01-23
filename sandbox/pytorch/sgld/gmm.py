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
from gmm_data_gen import genData

def sample_groups(state, y):
    m, log_s2, lw = state
    lam = []
    for yi in y:
        logprobs = lw + lpdf_normal(yi, m, log_s2)
        p = torch.exp(logprobs - logprobs.max())
        lam.append(torch.multinomial(p, 1).item())
    return lam
     
# logpdf of Normal
def lpdf_normal(x, m, log_v):
    return -(x - m) ** 2 / (2 * torch.exp(log_v)) - 0.5 * math.log(2 * math.pi) - 0.5 * log_v

def lpdf_loginvgamma_kernel(x, a, b):
    return -a * x - b * torch.exp(-x)

def loglike(yi, m, log_s2, logit_w):
    log_w = torch.log_softmax(logit_w, 0)
    return torch.logsumexp(log_w + lpdf_normal(yi, m, log_s2), 0)


def fit(y, J, nmcmc=1000, nburn=10, learning_rate=1e-3, L=50, prop_sd=1.0, seed=1, minibatch_size:int=0):
    # Set random seed for reproducibility
    torch.manual_seed(seed)

    # Define data type
    dtype = torch.float64

    # Define device-type (cpu / gpu)
    device = torch.device("cpu")

    # Number of obs
    N = len(y)

    def log_post(state):
        m, log_s2, lw = state
        
        if 0 < minibatch_size < N:
            idx = torch.randperm(N)[:minibatch_size]
            y_minibatch = y[idx]
        else:
            y_minibatch = y
        
        # log likelihood
        ll = torch.stack([loglike(yi, m, log_s2, lw) for yi in y_minibatch]).mean() * N

        # log prior
        lp_logsig2 = lpdf_loginvgamma_kernel(log_s2, 3, 2).sum()
        lp_mu = -(m ** 2 / 2).sum()
        lp_w = 0
        lp = lp_logsig2 + lp_mu + lp_w

        return ll + lp

    # Initialize parameters.
    mu = torch.randn(J, device=device, dtype=dtype) * 10
    mu.requires_grad = True

    log_sig2 = torch.empty(J, device=device, dtype=dtype).fill_(0)
    log_sig2.requires_grad = True

    logit_w = torch.empty(J, device=device, dtype=dtype).fill_(1 / J)
    logit_w.requires_grad = True

    state = [mu, log_sig2, logit_w]
    log_post_history = [log_post(state).item()]
    out = []

    for t in range(nmcmc + nburn):
        now = datetime.datetime.now()
        print('{} | iteration: {} / {} | normalized logpost: {}'.format(now, t + 1, nmcmc + nburn, log_post_history[-1] / N))

        sgld(log_post, state=state, log_post_history=log_post_history, eps=learning_rate) 

        if t >= nburn:
            out.append(copy.deepcopy(state))

        print('mu: {}'.format(mu.tolist()))
        print('sig2: {}'.format(torch.exp(log_sig2).tolist()))
        print('w: {}'.format(torch.softmax(logit_w, 0).tolist()))

        print()

    return {'chain': out, 'logpost_hist': log_post_history}
  

if __name__ == '__main__':
    data = genData(seed=1, nfactor=100)
    y = torch.tensor(data['y'])

    out = fit(y, J=5, L=100, learning_rate=1e-4, nmcmc=100, nburn=1000,
              minibatch_size=50, seed=3)

    ll = out['logpost_hist']

    # Retrieve labels
    labels = []
    for i in range(len(out['chain'])):
        state = out['chain'][i]
        labels.append( sample_groups(state, y) )
        print('\r{}/{}'.format(i, len(out['chain'])), end='')
    
    labels = np.array(labels)
    labels.mean(axis=0).round()


