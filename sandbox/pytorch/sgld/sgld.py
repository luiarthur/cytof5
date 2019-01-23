"""
HMC example
"""

import math
import torch

def sgld(log_post, state, log_post_history, eps=1e-4, dtype=torch.float64, verbose=0):
    """
    one step of an hmc

    log_post: function that takes `params` as argument only and returns log posterior
    state: list of torch parameters to take gradient of
    eps: learning rate for gradients

    details: http://people.ee.duke.edu/~lcarin/398_icmlpaper.pdf
    """
    # zero all the gradients
    def zero_grads(params):
        for param in params:
            if param.grad is not None:
                param.grad.zero_()

    # Compute gradients for each parameter
    def compute_gradients(params):
        zero_grads(params)
        loss = -log_post(params)
        log_post_history.append(-loss.item())
        loss.backward()

    # Compute gradient
    compute_gradients(state)

    # Make a half step for momentum at the beginning
    with torch.no_grad():
        for s in state:
            eta = torch.randn(s.size(), dtype=dtype) * eps
            s.data.sub_(s.grad.data * eps / 2 + eta)
