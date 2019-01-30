import torch
from torch.distributions import Normal

class VarParam():
    def __init__(self, size):
        m = torch.randn(size)
        m.requires_grad=True
        self.m = m

        log_s = torch.randn(size)
        log_s.requires_grad=True
        self.log_s= log_s

        self.size = size

    def sample(self):
        return torch.randn(self.size) * torch.exp(self.log_s) + self.m

    def log_prob(self, log_x):
        return Normal(self.m, torch.exp(self.log_s)).log_prob(log_x).sum()
