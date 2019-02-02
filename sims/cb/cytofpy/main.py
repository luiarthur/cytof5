import os
import torch
from readCB import readCB
from Cytof import Cytof
from simdata import simdata
import math
import matplotlib.pyplot as plt
import copy
import numpy as np
import pickle

def add_gridlines_Z(Z):
    J, K = Z.shape
    for j in range(J):
        plt.axhline(y=j+.5, color='grey', linewidth=.5)

    for k in range(K):
        plt.axvline(x=k+.5, color='grey', linewidth=.5)


if __name__ == '__main__':
    path_to_exp_results = 'results/test/'
    os.makedirs(path_to_exp_results, exist_ok=True)

    torch.manual_seed(1)
    np.random.seed(0)

    SIMULATE_DATA = True
    # SIMULATE_DATA = False
    cm_greys = plt.cm.get_cmap('Greys')
    
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
        # data = simdata(N=[30000, 10000, 20000], L0=3, L1=3, J=12, K=4)
        data = simdata(N=[3000, 3000, 3000], L0=3, L1=3, J=12, K=4)
        cb = data['data']
        plt.imshow(data['params']['Z'], aspect='auto', vmin=0, vmax=1, cmap=cm_greys)
        J, K = data['params']['Z'].shape
        add_gridlines_Z(data['params']['Z'])
        plt.savefig('{}/Z_true.pdf'.format(path_to_exp_results))
        plt.show()

    y = copy.deepcopy(cb['y'])

    plt.hist(y[0][:, 1], bins=100, density=True); plt.show()
    plt.hist(y[1][:, 3], bins=100, density=True); plt.show()
    plt.hist(y[2][:, -1], bins=100, density=True); plt.show()

    # Plot yi
    cm = plt.cm.get_cmap('bwr')
    cm.set_under(color='blue')
    cm.set_over(color='red')
    cm.set_bad(color='black')

    I = len(y)
    for i in range(I):
        plt.imshow(y[i], aspect='auto', vmin=-2, vmax=2, cmap=cm)
        plt.colorbar()
        plt.show()

    K = 4
    model = Cytof(data=cb, K=K, L=[3,3])
    priors = model.priors
    model = Cytof(data=cb, K=K, L=[3,3], priors=priors)
    # model.debug=True
    out = model.fit(data=cb, niters=1000, lr=1e-1, print_freq=1, eps=1e-6,
                    # minibatch_info={'prop': .1},
                    nmc=1)

    # Save output
    pickle.dump(out, open('{}/out.p'.format(path_to_exp_results), 'wb'))

    elbo = out['elbo']
    vp = out['vp']

    out = pickle.load(open('{}/out.p'.format(path_to_exp_results), 'rb'))

    plt.plot(elbo)
    plt.ylabel('ELBO / NSUM')
    plt.show()

    real_param_mean = {}
    for key in vp:
        if key != 'Z':
            real_param_mean[key] = vp[key].m
        else:
            real_param_mean[key] = vp[key].logit_p

    params = model.to_param_space(real_param_mean)
    # print(params)

    # for key in vp: print('{} log_s: {}'.format(key, (vp[key].log_s)))

    # Posterior Inference
    B = 100
    post = [model.to_param_space(model.sample_real_params(vp)) for b in range(B)]

    # Plot mu
    mu0 = torch.stack([p['mu0'].cumsum(2) for p in post]).reshape(B, model.L[0]).detach().numpy()
    mu1 = torch.stack([p['mu1'].cumsum(2) for p in post]).reshape(B, model.L[1]).detach().numpy()
    mu = np.concatenate((-mu0, mu1), 1)
    plt.boxplot(mu, showmeans=True, whis=[2.5, 97.5], showfliers=False)
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
        plt.boxplot(W[:, i, :], showmeans=True, whis=[2.5, 97.5], showfliers=False)
        plt.ylabel('$W_{}$'.format(i+1), rotation=0, labelpad=15)
        if SIMULATE_DATA:
            for yint in data['params']['W'][i, :].tolist():
                plt.axhline(yint)

    # plot v
    v = torch.stack([p['v'] for p in post]).detach().numpy().reshape(B, model.K)
    plt.subplot(model.I + 1, 1, model.I + 1)
    plt.boxplot(v.cumprod(1), showmeans=True, whis=[2.5, 97.5], showfliers=False)
    plt.ylabel('$v$', rotation=0, labelpad=15)
    if SIMULATE_DATA:
        for yint in data['params']['v'].tolist():
            plt.axhline(yint)

    plt.tight_layout()
    plt.show()

    # plot sig
    sig = torch.stack([p['sig'] for p in post]).detach().numpy()
    plt.boxplot(sig, showmeans=True, whis=[2.5, 97.5], showfliers=False)
    plt.xlabel('$\sigma$', fontsize=15)
    if SIMULATE_DATA:
        for yint in data['params']['sig'].tolist():
            plt.axhline(yint)

    plt.show()

    # Plot Z
    Z = torch.stack([p['Z'] for p in post]).detach().reshape((B, model.J, model.K)).numpy()
    plt.imshow(Z.mean(0) > .5, aspect='auto', vmin=0, vmax=1, cmap=cm_greys)
    add_gridlines_Z(Z.mean(0))
    plt.savefig('{}/Z.pdf'.format(path_to_exp_results))
    plt.show()

