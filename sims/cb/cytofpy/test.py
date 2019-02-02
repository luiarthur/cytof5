import torch
from torch.distributions import Normal
from Cytof import Cytof
from simdata import simdata

def mixLz(i, n, j, etaz, muz, sig, y):
    Lz = muz.size(2)
    assert(Lz == etaz.size(2))

    out = torch.empty(Lz)
    for l in range(Lz):
        out[l] = torch.log(etaz[i, j, l, 0]) + Normal(muz[0, 0, l, 0], sig[i]).log_prob(y[i][n, j, 0, 0])

    return out.logsumexp(0)

def mixJ(i, n, k, eta0, eta1, mu0, mu1, sig, Z, y):
    J = y[0].size(1)

    out = 0.0
    for j in range(J):
        zjk = Z[0, j, k]
        etaz = eta1 if zjk == 1 else eta0
        muz = mu1.cumsum(2) if zjk == 1 else -mu0.cumsum(2)
        out += mixLz(i, n, j, etaz, muz, sig, y) 
    return out

def mixK(i, n, eta0, eta1, mu0, mu1, sig, Z, W, y):
    K = Z.size(2)
    out = torch.empty(K)
    for k in range(K):
        out[k] = W[i, k].log() + mixJ(i, n, k, eta0, eta1, mu0, mu1, sig, Z, y)

    return out.logsumexp(0)

def naive_loglike(params, data, K, L, N):
    I = len(N)
    J = data['y'][0].size(1)
    y = data['y']
    Z = params['Z']
    W = params['W']
    eta0 = params['eta0']
    eta1 = params['eta1']
    mu0 = params['mu0']
    mu1 = params['mu1']
    sig = params['sig']
    Nsum = sum(N)

    ll = 0.0
    for i in range(I):
        for n in range(N[i]):
            ll += mixK(i, n, eta0, eta1, mu0, mu1, sig, Z, W, y) / Nsum
    return ll

if __name__ == '__main__':
    print('testing to see if I am broadcasting correctly...')

    torch.manual_seed(0)

    L0 = 3
    L1 = 4
    N = [100, 50, 80]
    J = 8
    K = 4
    L = [L0, L1]
    eps = 1e-6

    data = simdata(N=N, L0=L0, L1=L1, J=J, K=K)
    model = Cytof(data=data['data'], K=K, L=L)

    vp = model.init_vp()
    real_params = model.sample_real_params(vp)
    params = model.to_param_space(real_params)
    ll_model = model.loglike(real_params, data['data'])

    ll_naive = naive_loglike(params, data['data'], K, L, N)
    diff = ll_model - ll_naive
    print('ll_model: {} | ll_naive: {} | diff: {}'.format(ll_model, ll_naive, diff))
    assert(abs(diff / ll_model) < eps)
