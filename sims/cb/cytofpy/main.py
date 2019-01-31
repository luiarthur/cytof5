import torch
from readCB import readCB
from Cytof import Cytof
from simdata import simdata
import math
import matplotlib.pyplot as plt
import copy
import numpy as np

if __name__ == '__main__':
    torch.manual_seed(2)
    np.random.seed(0)

    SIMULATE_DATA = False

    if not SIMULATE_DATA:
        CB_FILEPATH = '../data/cb.txt'
        cb = readCB(CB_FILEPATH)
        cb['m'] = []
        tmp_J = 6
        for i in range(len(cb['y'])):
            cb['y'][i] = torch.tensor(cb['y'][i])[:, :tmp_J]
            cb['m'].append(torch.isnan(cb['y'][i]))
            # FIXME: missing values should be imputed
            cb['y'][i][cb['m'][i]] = torch.randn(cb['m'][i].sum()) * .5 - 5
    else:
        data = simdata(N=[3000, 1000, 2000])
        cb = data['data']

    y = copy.deepcopy(cb['y'])

    plt.hist(y[0][:, 1], bins=100, density=True); plt.show()
    plt.hist(y[1][:, 3], bins=100, density=True); plt.show()
    plt.hist(y[2][:, -1], bins=100, density=True); plt.show()

    # Plot yi
    # cm = plt.cm.get_cmap('bwr')
    # cm.set_under(color='blue')
    # cm.set_over(color='red')
    # cm.set_bad(color='black')
    # plt.imshow(y[0], aspect='auto', vmin=-2, vmax=2, cmap=cm)
    # plt.colorbar()
    # plt.show()

    model = Cytof(data=cb, K=10, L=[5,5])
    priors = model.priors
    priors['mu0'] = torch.distributions.Uniform(y[0].min(), 0)
    priors['mu1'] = torch.distributions.Uniform(0, y[1].max())
    model = Cytof(data=cb, K=10, L=[5,5], priors=priors)
    # model.debug=True
    out = model.fit(data=cb, niters=2000, lr=1e-2, print_freq=1, eps=1e-6,
                    minibatch_info={'prop': .1},
                    nmc=1)

    elbo = out['elbo']
    vp = out['vp']
    plt.plot(elbo)
    plt.ylabel('ELBO / NSUM')
    plt.show()

    real_param_mean = {}
    for key in vp:
        real_param_mean[key] = vp[key].m

    params = model.to_param_space(real_param_mean)
    # print(params)

    # for key in vp: print('{} log_s: {}'.format(key, (vp[key].log_s)))

    # Posterior Inference
    B = 100
    post = [model.to_param_space(model.sample_real_params(vp)) for b in range(B)]

    # Plot mu
    mu0 = torch.stack([p['mu0'] for p in post]).reshape(B, model.L[0]).detach().numpy()
    mu1 = torch.stack([p['mu1'] for p in post]).reshape(B, model.L[1]).detach().numpy()
    mu = np.concatenate((mu0, mu1), 1)
    plt.boxplot(mu)
    plt.ylabel('$\mu$', rotation=0)
    if SIMULATE_DATA:
        for yint in (data['params']['mu0'].tolist() + data['params']['mu1'].tolist()):
            plt.axhline(yint)

    plt.show()

    # plot W
    W = torch.stack([p['W'] for p in post]).detach().numpy()
    plt.figure()
    for i in range(model.I):
        plt.subplot(model.I + 1, 1, i + 1)
        plt.boxplot(W[:, i, :])
        plt.ylabel('$W_{}$'.format(i+1), rotation=0, labelpad=15)
        if SIMULATE_DATA:
            for yint in data['params']['W'][i, :].tolist():
                plt.axhline(yint)

    # plot v
    v = torch.stack([p['v'] for p in post]).detach().numpy().reshape(B, model.K)
    plt.subplot(model.I + 1, 1, model.I + 1)
    plt.boxplot(v)
    plt.ylabel('$v$', rotation=0, labelpad=15)
    if SIMULATE_DATA:
        for yint in data['params']['v'].tolist():
            plt.axhline(yint)

    plt.tight_layout()
    plt.show()

    # plot sig
    sig = torch.stack([p['sig'] for p in post]).detach().numpy()
    plt.boxplot(sig)
    plt.xlabel('$\sigma$', fontsize=15)
    if SIMULATE_DATA:
        for yint in data['params']['sig'].tolist():
            plt.axhline(yint)

    plt.show()

    # Simulate Z
