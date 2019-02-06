import copy
import numpy as np

import torch
from torch.distributions import Normal
from torch.distributions import Categorical

def sample_Z_lam(post, data):
    B = len(post)
    y = data['y']
    I = len(y)
    J = y[0].size(1)
    N = [yi.size(0) for yi in y]
    L0 = post[0]['mu0'].squeeze().numel()
    L1 = post[0]['mu1'].squeeze().numel()
    L = [L0, L1]
    K = post[0]['v'].squeeze().shape.numel()
    Z_init = np.random.binomial(1, .5, (J, K))
    lam_init = [np.random.choice(K, N[i]) for i in range(I)]

    state = {'Z': Z_init, 'lam': lam_init}
    chain = []

    def update_Zjk(j, k, state, b):
        lpj = [0.0, 0.0]
        for z in range(2):
            lpj[z] = torch.log(params['v'].squeeze()[k])
            for i in range(I):
                idx = np.argwhere(state['lam'][i] == k).squeeze()
                yij = y[i][idx, j].reshape((len(idx), 1))
                mukey = 'mu1' if z == 1 else 'mu0'
                etakey = 'eta1' if z == 1 else 'eta0'
                muz = post[b][mukey].reshape(1, L[z])
                sigi = post[b]['sig'][i]
                etaz_ij = post[b][etakey][i, j, :, 0].reshape(1, L[z])
                lpz_ij = torch.logsumexp(Normal(muz, sigi).log_prob(yij) +
                                         torch.log(etaz_ij), 1).sum()
                lpj[z] += lpz_ij
        lp = 1 / (1 + np.exp(lpj[0].item() - lpj[1].item()))
        zjk = 1 if lp > np.random.rand() else 0
        state['Z'][j, k] = zjk

    def update_Z(state, b):
        for j in range(J):
            for k in range(K):
                print(j, k)
                update_Zjk(j, k, state, b)

    def update_lam(state, b):
        NotImplemented

    for b in range(B):
        update_Z(state, b)
        update_lam(state, b)
        chain.append(copy.deepcopy(state))
