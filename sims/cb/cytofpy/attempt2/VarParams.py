import abc

import torch
from torch.distributions import Normal
from torch.distributions import Dirichlet
from torch.distributions import Gamma
from torch.distributions import Beta

class VarParam(abc.ABC):
    def __init__(self, size):
        self.size = size
        self.vp = None
    
    @abc.abstractmethod
    def sample(self):
        NotImplemented

    @abc.abstractmethod
    def logpdf(self, x):
        NotImplemented

class VPNormal(VarParam):
    def __init__(self, size):
        super().__init__(size)
        m = torch.randn(self.size)
        log_s = torch.randn(self.size)
        self.vp = torch.stack([m, log_s])
        self.vp.requires_grad = True

    def sample(self):
        return Normal(self.vp[0], self.vp[1].exp()).rsample()

    def logpdf(self, x):
        return Normal(self.vp[0], self.vp[1].exp()).log_prob(x)

class VPDirichlet(VarParam):
    def __init__(self, size):
        super().__init__(size)
        self.vp = torch.randn(size)
        self.vp.requires_grad = True

    def sample(self):
        return Dirichlet(self.vp.exp()).rsample()

    def logpdf(self, x):
        return Dirichlet(self.vp.exp()).log_prob(x)

class VPDirichletW(VarParam):
    def __init__(self, size):
        super().__init__(size)
        self.vp = torch.randn(size)
        self.vp.requires_grad = True
        self.I = size[0]
        self.K = size[1]

    def sample(self):
        out = torch.empty(self.size)
        for i in range(self.I):
            out[i, :] = Dirichlet(self.vp[i, :].exp()).rsample()
        return out

    def logpdf(self, x):
        out = 0.0
        for i in range(self.I):
            out += Dirichlet(self.vp[i, :].exp()).log_prob(x[i, :])
        return out

class VPDirichletEta(VarParam):
    def __init__(self, size):
        super().__init__(size)
        self.vp = torch.randn(size)
        self.vp.requires_grad = True
        self.I = size[0]
        self.J = size[1]
        self.Lz = size[2]

    def sample(self):
        out = torch.empty(self.size)
        for i in range(self.I):
            for j in range(self.J):
                out[i, j, :, 0] = Dirichlet(self.vp[i, j, :, 0].exp().squeeze()).rsample()
        return out.reshape(self.I, self.J, self.Lz, 1)

    def logpdf(self, x):
        out = 0.0
        for i in range(self.I):
            for j in range(self.J):
                a = self.vp[i, j, :, 0].exp().squeeze()
                out += Dirichlet(a).log_prob(x[i, j, :, 0].squeeze())
        return out

class VPGamma(VarParam):
    def __init__(self, size):
        super().__init__(size)
        concentration = torch.randn(size)
        rate = torch.randn(size)
        self.vp = torch.stack([concentration, rate])
        self.vp.requires_grad = True

    def sample(self):
        return Gamma(self.vp[0].exp(), self.vp[1].exp()).rsample()

    def logpdf(self, x):
        return Gamma(self.vp[0].exp(), self.vp[1].exp()).log_prob(x)

class VPBeta(VarParam):
    def __init__(self, size):
        super().__init__(size)
        a = torch.randn(size)
        b = torch.randn(size)
        self.vp = torch.stack([a, b])
        self.vp.requires_grad = True

    def sample(self):
        return Beta(self.vp[0].exp(), self.vp[1].exp()).rsample()

    def logpdf(self, x):
        return Beta(self.vp[0].exp(), self.vp[1].exp()).log_prob(x)

