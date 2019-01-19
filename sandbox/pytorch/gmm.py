# https://www.kaggle.com/aakashns/pytorch-basics-linear-regression-from-scratch
# https://angusturner.github.io/generative_models/2017/11/03/pytorch-gaussian-mixture-model.html
# https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html

import torch
import time
import math
from gmm_data_gen import genData

# Set random seed for reproducibility
torch.manual_seed(1)

# Set number of cpus to use
torch.set_num_threads(4)

# Define data type
dtype = torch.float64

# Define device-type (cpu / gpu)
device = torch.device("cpu")

# Param dimensions
J = 3
data = genData()
y_data = torch.tensor(data['y'])
y_mean = torch.mean(y_data).item()
y_sd = torch.std(y_data).item()
y_cs = (y_data - y_mean) / y_sd
N = len(y_cs)

# Create random Tensors for weights.
mu = torch.randn(J, device=device, dtype=dtype)
mu.requires_grad=True
log_sig2 = torch.empty(J, device=device, dtype=dtype).fill_(-5)
log_sig2.requires_grad=True
logit_w = torch.empty(J, device=device, dtype=dtype).fill_(1 / J)
logit_w.requires_grad=True

# logpdf of Normal
def lpdf_normal(x, m, v):
    return -(x - m) ** 2 / (2 * v) - 0.5 * torch.log(2 * math.pi * v)

def pdf_normal(x, m, v):
    return torch.exp(lpdf_normal(x, m, v))

def lpdf_loginvgamma_kernel(x, a, b):
    return -a * x - b * torch.exp(-x)

def loglike(yi, m, log_s2, logit_w):
    s2 = torch.exp(log_s2)
    log_w = torch.log_softmax(logit_w, 0)
    return torch.logsumexp(log_w + lpdf_normal(yi, m, s2), 0)
    # which is equivalent to and more numerically stable to:
    # w = torch.softmax(logit_w, 0)
    # return torch.log(w.dot(pdf_normal(yi, mu, sig2)))

# loglike(y_data[0], mu, log_sig2, logit_w)

learning_rate = 1e-3
eps = 1E-8
optimizer = torch.optim.Adam([mu, log_sig2, logit_w], lr=learning_rate)
ll_out = [-math.inf, ]

for t in range(100000):
    # zero out the gradient
    optimizer.zero_grad()

    # Forward pass
    ll = torch.stack([loglike(yi, mu, log_sig2, logit_w) for yi in y_cs]).sum()
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

    print("{}: loglike: {}".format(t, ll.item() / N))
    print('mu: {}'.format(list(map(lambda m: m * y_sd + y_mean, mu.tolist()))))
    print('sig2: {}'.format(list(map(lambda s2: s2 * y_sd * y_sd, torch.exp(log_sig2).tolist()))))
    print('w: {}'.format(torch.softmax(logit_w, 0).tolist()))

    # Use autograd to compute the backward pass. 
    loss.backward()

    # Update weights
    optimizer.step()

    # SAME AS ABOVE.
    #
    # Update weights using gradient descent.
    # with torch.no_grad():
    #     mu -= mu.grad * learning_rate
    #     log_sig2 -= log_sig2.grad * learning_rate
    #     logit_w -= logit_w.grad * learning_rate
    #
    #     # Manually zero the gradients after updating weights
    #     mu.grad.zero_()
    #     log_sig2.grad.zero_()
    #     logit_w.grad.zero_()
  

