import torch
from torch.distributions import Normal

class VarParam():
    def __init__(self, size, init_m=None, init_log_s=None):
        if init_m is None:
            m = torch.zeros(size)
        else:
            m = torch.zeros(size) + init_m
        m.requires_grad=True
        self.m = m

        if init_log_s is None:
            log_s = torch.zeros(size)
        else:
            log_s = torch.zeros(size) + init_log_s
        log_s.requires_grad=True
        self.log_s= log_s

        self.size = size

    def sample(self):
        return torch.randn(self.size) * torch.exp(self.log_s) + self.m

    def log_prob(self, x):
        return Normal(self.m, torch.exp(self.log_s)).log_prob(x).sum()

