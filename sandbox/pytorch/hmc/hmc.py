"""
HMC example
"""

import math
import torch

def hmc(log_post, state, L=50, eps=1e-4, prop_sd=1.0, dtype=torch.float64):
    """
    one step of an hmc

    state: list of torch parameters to take gradient of
    log_post: function that takes `params` as argument only and returns log posterior
    L: leap-frog steps
    eps: learning rate for gradients
    prop_sd: proposal standard deviation

    details: https://arxiv.org/pdf/1206.1901.pdf
    """
    # zero all the gradients
    def zero_grads(params):
        for param in params:
            if param.grad is not None:
                param.grad.zero_()

    # Turn off gradient tracker
    def grad_off(params):
        for param in params:
            param.requires_grad = False

    # Turn on gradient tracker
    def grad_on(params):
        for param in params:
            param.requires_grad = True

    # Compute gradients for each parameter
    def compute_gradients(params):
        zero_grads(params)
        loss = -log_post(params)
        loss.backward()

    # proposed params
    grad_off(state)
    q = [s.clone() for s in state]
    p = [torch.randn(qj.size(), dtype=dtype) * prop_sd for qj in q]
    current_p = p

    # Compute gradient
    grad_on(q)
    compute_gradients(q)

    # Make a half step for momentum at the beginning
    with torch.no_grad():
        for (pj, qj) in zip(p, q):
            pj.data.sub_(eps * qj.grad.data / 2)

    for i in range(L):
        print('\r{} / {}'.format(i, L), end='')
        compute_gradients(q)

        with torch.no_grad():
            for (pj, qj) in zip(p, q):
                # make a full step for position
                qj.data.add_(eps * pj.data)

                if i < L - 1:
                    # Make a full step for momentum, except at the end
                    pj.data.sub_(eps * qj.grad.data)
                else:
                    # Make a half step for momentum at the end.
                    pj.data.sub_(eps * qj.grad.data / 2)

    print()

    # Negate momentum at the end
    compute_gradients(q)
    with torch.no_grad():
        for pj in p:
            pj = -pj

    # Evaluate potential and kinetic energies at start and end of trajectory
    curr_U = -log_post(state).item()
    curr_K = sum([(pj @ pj).item() for pj in current_p]) / 2
    prop_U = -log_post(q).item()
    prop_K = sum([(pj @ pj).item() for pj in p]) / 2

    # print(curr_U, prop_U, curr_K, prop_K)
    # print(q)

    # Compute acceptance prob
    log_acc_prob = curr_U + curr_K - prop_U - prop_K
    print('log acceptance prob: {}'.format(log_acc_prob))

    grad_on(state)
    if log_acc_prob > torch.log(torch.rand(1)).item():
        print("ACCEPT!")
        return q
    else:
        return state

