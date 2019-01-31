import torch

from torch.distributions import Dirichlet
from torch.distributions import Categorical
from torch.distributions import Bernoulli
from torch.distributions import Beta
from torch.distributions import Normal
from torch.distributions import Uniform
from torch.distributions import Gamma
from torch.distributions.log_normal import LogNormal

def simdata(N=[300, 100, 200], J=25, K=10, L0=5, L1=5, alpha=10):
    I = len(N)
    data = {'y': [], 'm': []}

    a_W = torch.ones(K) * 3 / K
    a_eta0 = torch.ones(L0)
    a_eta1 = torch.ones(L1)

    W = Dirichlet(a_W).sample((I, ))
    v = Beta(alpha / K, 1).sample((K, ))
    eta0 = Dirichlet(a_eta0).sample((I, J))
    eta1 = Dirichlet(a_eta1).sample((I, J))

    mu0 = Uniform(-5, 0).sample((L0, ))
    mu1 = Uniform(0, 5).sample((L1, ))
    sig = Gamma(1, 3).sample((I, ))

    Z = Bernoulli(v).sample((J, )).reshape(J, K)

    params = {'W': W, 'v': v, 'eta0': eta0, 'eta1': eta1,
              'mu0': mu0, 'mu1': mu1, 'sig': sig, 'Z': Z}

    lam = []
    for i in range(I):
        wi = Dirichlet(a_W).sample()

        gam0i = []
        gam1i = []
        for j in range(J):
            gam0i.append(Categorical(eta0[i, j, :]).sample((N[i], )))
            gam1i.append(Categorical(eta1[i, j, :]).sample((N[i], )))

        gam0i = torch.stack(gam0i, 1)
        gam1i = torch.stack(gam1i, 1)

        lami = Categorical(wi).sample((N[i], ))
        lam.append(lami)

        Zi = Z[:, lami]
        Zi.transpose_(0, 1)

        mui = Zi * mu0[gam0i] + (1 - Zi) * (mu1[gam1i])

        yi = Normal(mui, sig[i]).sample()

        data['y'].append(yi + 0)
        data['m'].append(yi + 0)

    params['lam'] = lam

    return {'data': data, 'params': params}

# data = simdata()
