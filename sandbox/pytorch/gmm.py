# https://www.kaggle.com/aakashns/pytorch-basics-linear-regression-from-scratch
# https://angusturner.github.io/generative_models/2017/11/03/pytorch-gaussian-mixture-model.html
# https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html

import torch
import math
from gmm_data_gen import genData

# Define data type
dtype = torch.float64

# Define device-type (cpu / gpu)
device = torch.device("cpu")

# Param dimensions
J = 3
data = genData()
y_data = torch.tensor(data['y'])
N = len(y_data)

# Create random Tensors to hold input and outputs.
y = 1 # torch.randn(1, device=device, dtype=dtype)

# Create random Tensors for weights.
mu = torch.randn(J, device=device, dtype=dtype, requires_grad=True)
sig2 = torch.rand(J, device=device, dtype=dtype, requires_grad=True)
w = torch.ones(J, device=device, dtype=dtype, requires_grad=True) / J

# logpdf of Normal
def lpdf_normal(x, m, v):
    return -(x - m) ** 2 / (2 * v) - 0.5 * torch.log(2 * math.pi * v)

def pdf_normal(x, m, v):
    return torch.exp(lpdf_normal(x, m, v))

def loglike(yi, m, s2, w):
    return w @ pdf_normal(yi, mu, sig2)
    # SAME AS:
    # return w.dot(pdf_normal(yi, mu, sig2))

learning_rate = 1e-6
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for t in range(500):
    # Forward pass
    ll = torch.stack([loglike(yi, mu, sig2, w) for yi in y_data])

    # Compute and print loss using operations on Tensors.
    loss = -ll.sum()
    print(t, loss.item())

    # Use autograd to compute the backward pass. 
    loss.backward()

    # Update weights using gradient descent
    with torch.no_grad():
        mu -= learning_rate * mu.grad
        sig2 -= learning_rate * sig2.grad
        w -= learning_rate * w.grad

        # Manually zero the gradients after updating weights
        mu.grad.zero_()
        sig2.grad.zero_()
        w.grad.zero_()

