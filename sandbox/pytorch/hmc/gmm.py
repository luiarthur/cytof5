import torch
import time
import math
import copy
import datetime

from hmc import hmc
import sys
sys.path.append('..')
from gmm_data_gen import genData

# logpdf of Normal
def lpdf_normal(x, m, v):
    return -(x - m) ** 2 / (2 * v) - 0.5 * torch.log(2 * math.pi * v)

def lpdf_loginvgamma_kernel(x, a, b):
    return -a * x - b * torch.exp(-x)

def loglike(yi, m, log_s2, logit_w):
    s2 = torch.exp(log_s2)
    log_w = torch.log_softmax(logit_w, 0)
    return torch.logsumexp(log_w + lpdf_normal(yi, m, s2), 0)


def fit(y, J, nmcmc=1000, nburn=10, learning_rate=1e-3, L=50, prop_sd=1.0, seed=1):
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

        # log likelihood
        ll = torch.stack([loglike(yi, m, log_s2, lw) for yi in y]).sum()

        # log prior
        lp_logsig2 = lpdf_loginvgamma_kernel(log_s2, 3, 2).sum()
        lp_mu = -(m ** 2 / 2).sum()
        lp_w = 0
        lp = lp_logsig2 + lp_mu + lp_w

        return ll + lp

    # Initialize parameters.
    mu = torch.randn(J, device=device, dtype=dtype) # * 10
    mu.requires_grad = True

    log_sig2 = torch.empty(J, device=device, dtype=dtype).fill_(-3)
    log_sig2.requires_grad = True

    logit_w = torch.empty(J, device=device, dtype=dtype).fill_(1 / J)
    logit_w.requires_grad = True

    state = [mu, log_sig2, logit_w]
    log_post_history = [log_post(state).item() / N]
    out = []

    for t in range(nmcmc + nburn):
        now = datetime.datetime.now()
        print('{} | iteration: {} / {} | normalized logpost: {}'.format(now, t + 1, nmcmc, log_post_history[-1]))

        hmc(log_post, state=state, log_post_history=log_post_history,
            L=L, eps=learning_rate, prop_sd=prop_sd) 
        log_post_history[-1] /= N

        if t >= nburn:
            out.append(copy.deepcopy(state))

        print('mu: {}'.format(mu.tolist()))
        print('sig2: {}'.format(torch.exp(log_sig2).tolist()))
        print('w: {}'.format(torch.softmax(logit_w, 0).tolist()))

        print()

    return {'chain': out, 'logpost_hist': log_post_history}
  

if __name__ == '__main__':
    data = genData()
    y = torch.tensor(data['y'])

    out = fit(y, J=3, L=50, learning_rate=1e-3, nmcmc=100, nburn=100)

