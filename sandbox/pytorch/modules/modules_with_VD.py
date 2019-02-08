import abc
import matplotlib.pyplot as plt

import torch
from torch.distributions import Normal, Gamma
from torch.distributions.kl import kl_divergence as kld
from torch.nn import Parameter as Param

# VD: Variational Distribution
# VP: Variational Parameters
# VI: Variational Inference

class VD(abc.ABC):
    @abc.abstractmethod
    def __init__(self, size):
        NotImplemented

    @abc.abstractmethod
    def dist(self):
        NotImplemented

    def rsample(self, n=torch.Size([])):
        return self.dist().rsample(n)

    def log_prob(self):
        return self.dist().log_prob()


class VDGamma(VD):
    def __init__(self, size):
        log_conc = torch.randn(size)
        log_rate = torch.randn(size)

        self.vp = Param(torch.stack([log_conc, log_rate]), requires_grad=True)
        self.size = size

    def dist(self):
        return Gamma(self.vp[0].exp(), self.vp[1].exp())


class VDNormal(VD):
    def __init__(self, size):
        m = torch.randn(size)
        log_s = torch.randn(size)

        self.vp = Param(torch.stack([m, log_s]), requires_grad=True)
        self.size = size

    def dist(self):
        return Normal(self.vp[0], self.vp[1].exp())


class VI(torch.nn.Module):
    def __init__(self):
        # Call parent's init
        super(VI, self).__init__()

        # Register variational parameters
        for key in self.__dict__:
            param = self.__getattribute__(key)
            if issubclass(type(param), VD):
                self.__setattr__(key + '_vp', param.vp)


class LinReg(VI):
    def __init__(self, p):
        # Assign variational distributions
        self.b = VDNormal(p)
        self.sig = VDGamma(1)

        # This must be done after assigning variational distributions
        super(LinReg, self).__init__()


    def forward(self, y, X):
        # Sample parameters
        b = self.b.rsample()
        sig = self.sig.rsample()

        # Compute loglike
        ll = Normal(X.matmul(b), sig).log_prob(y)

        # Compute kl_qp
        kl_qp = kld(self.b.dist(), Normal(0, 1)) + kld(self.sig.dist(), Gamma(1, 1))

        # Compute ELBO
        elbo = ll.sum() - kl_qp.sum()

        return elbo


if __name__ == '__main__':
    torch.manual_seed(0)
    N = 1000
    b = torch.tensor([3.0, -2.0])
    p = b.size().numel()
    sig = torch.tensor(.5)

    X = torch.cat((torch.ones(N, 1), torch.randn(N, p - 1)), 1)
    y = X.matmul(b) + torch.randn(N) * sig
    model = LinReg(p)

    elbo_hist = []
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)
    max_iter = 1000

    for t in range(max_iter):
        elbo = model(y, X)
        loss = -elbo / N
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        elbo_hist.append(-loss.item())
        if t % 100 == 0:
            print('iteration: {} / {} | elbo: {}'.format(t, max_iter, elbo_hist[-1]))

        if t > 10 and abs(elbo_hist[-1] / elbo_hist[-2] - 1) < 1e-4:
            print('Convergence suspected! Ending optimizer early.')
            break

    # print(model.state_dict())
    print(model.b.dist().mean, model.sig.dist().mean)
    print(b, sig)

    plt.plot(elbo_hist); plt.show()
