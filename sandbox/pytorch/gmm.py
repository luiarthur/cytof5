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
N = len(y_data)

# Create random Tensors for weights.
mu = torch.randn(J, device=device, dtype=dtype, requires_grad=True)
log_sig2 = torch.zeros(J, device=device, dtype=dtype, requires_grad=True)
logit_w = torch.randn(J, device=device, dtype=dtype, requires_grad=True)

# logpdf of Normal
def lpdf_normal(x, m, v):
    return -(x - m) ** 2 / (2 * v) - 0.5 * torch.log(2 * math.pi * v)

def pdf_normal(x, m, v):
    return torch.exp(lpdf_normal(x, m, v))

def lpdf_loginvgamma_kernel(x, a, b):
    return -a * x - b * torch.exp(-x)

def loglike(yi, m, log_s2, logit_w):
    sig2 = torch.exp(log_s2)
    w = torch.softmax(logit_w, 0)
    return torch.log(w.dot(pdf_normal(yi, mu, sig2)))

# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

learning_rate = 1e-3
for t in range(10000):
    # time.sleep(1)

    # Forward pass
    ll = torch.stack([loglike(yi, mu, log_sig2, logit_w) for yi in y_data]).sum()
    lp_logsig2 = lpdf_loginvgamma_kernel(log_sig2, 3, 2).sum()
    lp_logit_w = 0 # TODO
    lp = lp_logsig2 + lp_logit_w
    #
    # Compute and print loss using operations on Tensors.
    log_post = ll + lp
    loss = -(log_post) / N
    print("{}: loglike: {}".format(t, ll.item() / N))
    print('mu: {}'.format(mu.tolist()))
    print('sig2: {}'.format(torch.exp(log_sig2).tolist()))
    print('w: {}'.format(torch.softmax(logit_w, 0).tolist()))
    #
    # Use autograd to compute the backward pass. 
    loss.backward()
    #
    # Update weights using gradient descent
    with torch.no_grad():
        mu -= mu.grad * learning_rate
        log_sig2 -= log_sig2.grad * learning_rate
        logit_w -= logit_w.grad * learning_rate
        #
        # Manually zero the gradients after updating weights
        mu.grad.zero_()
        log_sig2.grad.zero_()
        logit_w.grad.zero_()


