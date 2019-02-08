import abc

import torch
from torch.distributions import Normal
from torch.distributions.log_normal import LogNormal
from torch.distributions import Dirichlet
from torch.distributions import Gamma
from torch.distributions import Beta

class VarDist(abc.ABC):
    def __init__(self, size):
        self.size = size
        self.vp = None
    
    def rsample(self):
        return self.dist().rsample()

    def log_prob(self, x):
        return self.dist().log_prob(x)

    @abc.abstractmethod
    def dist(self):
        NotImplemented


class VDNormal(VarDist):
    def __init__(self, size):
        super().__init__(size)
        m = torch.randn(self.size)
        log_s = torch.randn(self.size)
        self.vp = torch.stack([m, log_s])
        self.vp.requires_grad = True

    def dist(self):
        return Normal(self.vp[0], self.vp[1].exp())

class VDDirichlet(VarDist):
    def __init__(self, size):
        super().__init__(size)
        self.vp = torch.randn(size)
        self.vp.requires_grad = True

    def dist(self):
        return Dirichlet(self.vp.exp())

class VDGamma(VarDist):
    def __init__(self, size):
        super().__init__(size)
        concentration = torch.randn(size)
        rate = torch.randn(size)
        self.vp = torch.stack([concentration, rate])
        self.vp.requires_grad = True

    def dist(self):
        return Gamma(self.vp[0].exp(), self.vp[1].exp())

class VDBeta(VarDist):
    def __init__(self, size):
        super().__init__(size)
        a = torch.randn(size)
        b = torch.randn(size)
        self.vp = torch.stack([a, b])
        self.vp.requires_grad = True

    def dist(self):
        return Beta(self.vp[0].exp(), self.vp[1].exp())

class VDLogNormal(VarDist):
    def __init__(self, size):
        super().__init__(size)
        m = torch.randn(size)
        s = torch.randn(size)
        self.vp = torch.stack([m, s])
        self.vp.requires_grad = True

    def dist(self):
        return LogNormal(self.vp[0], self.vp[1].exp())

