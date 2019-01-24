import torch

# Transforms from support space to Real space
def logit(p, a=0.0, b=1.0):
    """
    for scalar parameters with bounded support (no gaps)
    basically a logit transform
    """
    p_scaled = (p - a) / (b - a)
    return torch.log(p_scaled) - torch.log(1.0 - p_scaled)

def invsoftmax(p):
    """
    Basically for transforming a Dirichlet to real space
    """
    return torch.log(p) - torch.log(p.max())

# Transforms from to Real space to support space
def invlogit(x, a=0.0, b=1.0):
    """
    sigmoid
    """
    u = torch.sigmoid(x) 
    return (b - a) * u + a

### Density transformations
def lpdf_logx(logx, lpdf_x):
    x = torch.exp(logx)
    return lpdf_x(x) + logx

def lpdf_logitx(logitx, lpdf_x, a=0.0, b=1.0):
    x = invlogit(logitx, a, b)
    return lpdf_x(x) + torch.log(b - a) + logitx - 2 * torch.log(1 + torch.exp(logitx))

def lpdf_real_dirichlet(r, lpdf_p):
    """
    Remember to perform: r -= r.max() when before zeroing out the gradient
    """
    K = p.size().numel
    J = torch.empty([K, K])
    p = real_to_simplex(r) 

    sum_x = torch.exp(x).sum()
    for i in range(K):
        for j in range(i + 1):
            if i == j:
                # J[i, j] = torch.exp(x[i]) * (sum_x - torch.exp(x[j])) / (sum_x ** 2)
                J[i, j] = p[i] * (1 - p[i])
            else:
                # tmp = torch.exp(x[i] + x[j]) / (sum_x ** 2)
                tmp = -p[i] * p[j]
                J[i, j] = tmp
                J[j, i] = tmp

    return lpdf_p(p) + torch.abs(torch.logdet(J))
