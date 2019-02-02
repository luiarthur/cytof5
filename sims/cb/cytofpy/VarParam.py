import torch
from torch.distributions import Normal
from torch.distributions import Bernoulli

class VarParam():
    def __init__(self, size, init_m=None, init_log_s=None):
        if init_m is None:
            m = torch.randn(size)
        else:
            m = torch.ones(size) * init_m
        m.requires_grad=True
        self.m = m

        if init_log_s is None:
            log_s = torch.randn(size)
        else:
            log_s = torch.ones(size) * init_log_s
        log_s.requires_grad=True
        self.log_s= log_s

        self.size = size

    def sample(self):
        return torch.randn(self.size) * torch.exp(self.log_s) + self.m

    def log_prob(self, x):
        return Normal(self.m, torch.exp(self.log_s)).log_prob(x).sum()

class VarParamBernoulli():
    def __init__(self, size):
        logit_p = torch.randn(size)
        logit_p.requires_grad=True
        self.logit_p = logit_p

    def sample(self):
        p = torch.sigmoid(self.logit_p)
        return Bernoulli(p).sample()

    def log_prob(self, z):
        p = torch.sigmoid(self.logit_p)
        return Bernoulli(p).log_prob(z).sum()


