# https://www.kaggle.com/aakashns/pytorch-basics-linear-regression-from-scratch
# https://angusturner.github.io/generative_models/2017/11/03/pytorch-gaussian-mixture-model.html
# https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
# https://pytorch.org/tutorials/beginner/examples_nn/two_layer_net_module.html

import torch
import time
import math

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

def fit(y, J, max_iter, eps=1e-8, learning_rate=1e-3, momentum=.9, use_sgd=False, seed=1):
    # Set random seed for reproducibility
    torch.manual_seed(seed)

    # Define data type
    dtype = torch.float64

    # Define device-type (cpu / gpu)
    device = torch.device("cpu")

    # Number of obs
    N = len(y)

    # Initialize parameters.
    mu = torch.randn(J, device=device, dtype=dtype) * 10
    mu.requires_grad = True

    log_sig2 = torch.empty(J, device=device, dtype=dtype).fill_(-5)
    log_sig2.requires_grad = True

    logit_w = torch.empty(J, device=device, dtype=dtype).fill_(1 / J)
    logit_w.requires_grad = True

    # Optimizer
    if use_sgd:
        optimizer = torch.optim.Adam([mu, log_sig2, logit_w], lr=learning_rate)
    else:
        optimizer = torch.optim.SGD([mu, log_sig2, logit_w],
                                    lr=learning_rate, momentum=momentum)

    # loglikelihood tracker
    ll_out = [-math.inf]

    def get_lr():
        return optimizer.param_groups[0]['lr']

    def set_lr(lr):
        optimizer.param_groups[0]['lr'] = lr

    period = 5

    for t in range(max_iter):
        # zero out the gradient
        optimizer.zero_grad()
        if t % period == 0 and get_lr() > 1e-6 and t > 0:
            period *= 2
            set_lr(get_lr() / 2)

        # Forward pass
        ll = torch.stack([loglike(yi, mu, log_sig2, logit_w) for yi in y]).sum()
        ll_out.append(ll.item())
        lp_logsig2 = lpdf_loginvgamma_kernel(log_sig2, 3, 2).sum()
        lp_logit_w = 0 # TODO
        lp = lp_logsig2 + lp_logit_w

        # Compute and print loss using operations on Tensors.
        log_post = ll + lp
        loss = -(log_post) / N
        ll_diff = ll_out[-1] - ll_out[-2]

        if ll_diff / N < eps:
            break
        else:
            print('ll mean improvement: {}'.format(ll_diff / N))

        print('lr: {}'.format(get_lr()))
        print("{}: loglike: {}".format(t, ll.item() / N))
        print('mu: {}'.format(mu.tolist()))
        print('sig2: {}'.format(torch.exp(log_sig2).tolist()))
        print('w: {}'.format(torch.softmax(logit_w, 0).tolist()))

        # Use autograd to compute the backward pass. 
        loss.backward()

        # Update weights
        optimizer.step()

    params = {'mu': mu, 'sig2': torch.exp(log_sig2), 'w': torch.softmax(logit_w, 0)}
    return {'params': params, 'll': ll}
  

if __name__ == '__main__':
    data = genData()
    out = fit(torch.tensor(data['y']), J=3, max_iter=100000, learning_rate=1e-3, seed=1)
