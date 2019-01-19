import torch
import time
import math

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

def fit(y, J, max_iter=1000, learning_rate=1e-3, L=50, prop_sd=1.0, seed=1):
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
        lp_mu = (-m**2 / 2).sum()
        lp = lp_logsig2 + lp_mu

        return ll + lp

    # Initialize parameters.
    mu = torch.randn(J, device=device, dtype=dtype) * 10
    mu.requires_grad = True

    log_sig2 = torch.empty(J, device=device, dtype=dtype).fill_(-3)
    log_sig2.requires_grad = True

    logit_w = torch.empty(J, device=device, dtype=dtype).fill_(1 / J)
    logit_w.requires_grad = True

    state = [mu, log_sig2, logit_w]

    lpost = []
    for t in range(max_iter):
        lpost.append(log_post(state).item())
        print('iteration: {} / {} -- logpost: {}'.format(t, max_iter, lpost[-1]))
        state = hmc(log_post, state=state, L=L, eps=learning_rate, prop_sd=prop_sd) 

        print('mu: {}'.format(state[0].tolist()))
        print('sig2: {}'.format(torch.exp(state[1]).tolist()))
        print('w: {}'.format(torch.softmax(state[2], 0).tolist()))

    return {'params': params, 'll': ll}
  

if __name__ == '__main__':
    data = genData()
    y = torch.tensor(data['y'])

    out = fit(y, J=3, L=10, learning_rate=1e-3)

