import torch

def elbo(state, lr=1e-4, dtype=torch.float64, verbose=0):
    # return a float
    # use my sgld implementation as a reference
    pass

def elbo_mean(state, elbo_history, nsamps=10, lr=1e-4, dtype=torch.float64, verbose=0):
    pass


