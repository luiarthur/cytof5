import os
import torch
from readCB import readCB
from Cytof import Cytof
from simdata import simdata
from torch.distributions import Normal
import math
import matplotlib.pyplot as plt
import copy
import numpy as np
import pickle

torch.set_default_dtype(torch.float64)

def add_gridlines_Z(Z):
    J, K = Z.shape
    for j in range(J):
        plt.axhline(y=j+.5, color='grey', linewidth=.5)

    for k in range(K):
        plt.axvline(x=k+.5, color='grey', linewidth=.5)


if __name__ == '__main__':
    sbt = torch.distributions.StickBreakingTransform()
    path_to_exp_results = 'results/test/'
    os.makedirs(path_to_exp_results, exist_ok=True)

    torch.manual_seed(0)
    np.random.seed(0)

    SIMULATE_DATA = True
    # SIMULATE_DATA = False
    cm_greys = plt.cm.get_cmap('Greys')
    
    if not SIMULATE_DATA:
        CB_FILEPATH = '../../data/cb.txt'
        cb = readCB(CB_FILEPATH)
        cb['m'] = []
        tmp_J = 25
        for i in range(len(cb['y'])):
            cb['y'][i] = torch.tensor(cb['y'][i])[:, :tmp_J]
            cb['m'].append(torch.isnan(cb['y'][i]))
            # FIXME: missing values should be imputed
            cb['y'][i][cb['m'][i]] = torch.randn(cb['m'][i].sum()) * .5 - 5
    else:
        # data = simdata(N=[30000, 10000, 20000], L0=3, L1=3, J=12, K=4)
        # data = simdata(N=[3000, 3000, 3000], L0=3, L1=3, J=12, K=4)
        data = simdata(N=[30000, 10000, 20000], L0=1, L1=1, J=4, a_W=[300, 700])
        # data = simdata(N=[30000, 10000, 20000], L0=3, L1=3, J=24, a_W=[300, 200, 500])
        cb = data['data']
        plt.imshow(data['params']['Z'], aspect='auto', vmin=0, vmax=1, cmap=cm_greys)
        J, K = data['params']['Z'].shape
        add_gridlines_Z(data['params']['Z'])
        plt.savefig('{}/Z_true.pdf'.format(path_to_exp_results))
        plt.show()

    y = copy.deepcopy(cb['y'])

    plt.hist(y[0][:, 1], bins=100, density=True); plt.xlim(-20, 20); plt.show()
    plt.hist(y[1][:, 3], bins=100, density=True); plt.xlim(-20, 20); plt.show()
    plt.hist(y[2][:, -1], bins=100, density=True); plt.xlim(-20, 20); plt.show()

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

    K = 10
    L = [2, 2]
    model = Cytof(data=cb, K=K, L=L)
    priors = model.priors
    model = Cytof(data=cb, K=K, L=L, priors=priors)
    model.debug=0
    out = model.fit(data=cb, niters=3000, lr=1e-1, print_freq=10, eps=0,
                    minibatch_info={'prop': .01},
                    nmc=1, seed=10)

    # Save output
    pickle.dump(out, open('{}/out.p'.format(path_to_exp_results), 'wb'))

    elbo = out['elbo']
    vp = out['vp']

    # out = pickle.load(open('{}/out.p'.format(path_to_exp_results), 'rb'))

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
    mu0 = torch.stack([p['mu0'].cumsum(0) for p in post]).detach().numpy()
    mu1 = torch.stack([p['mu1'].cumsum(0) for p in post]).detach().numpy()
    mu = np.concatenate((-(model.iota + mu0), model.iota + mu1), 1)
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
    v = torch.stack([p['v'] for p in post]).detach().numpy()
    plt.subplot(model.I + 1, 1, model.I + 1)
    plt.boxplot(v.cumprod(1), showmeans=True, whis=[2.5, 97.5], showfliers=False)
    plt.ylabel('$v$', rotation=0, labelpad=15)
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

    # plot alpha
    alpha = torch.stack([p['alpha'] for p in post]).detach().numpy().squeeze()
    plt.hist(alpha)
    plt.xlabel('alpha', fontsize=15)
    plt.show()


    # Plot Z
    # Z = torch.stack([p['Z'] for p in post]).detach().reshape((B, model.J, model.K)).numpy()
    H = torch.stack([p['H'] for p in post]).detach()
    v = torch.stack([p['v'] for p in post]).detach()
    Z = v.log().cumsum(1)[:, None, :] > Normal(0, 1).cdf(H).log()
    Z = Z.numpy()
    plt.imshow(Z.mean(0) > .5, aspect='auto', vmin=0, vmax=1, cmap=cm_greys)
    # plt.imshow(Z.mean(0), aspect='auto', vmin=0, vmax=1, cmap=cm_greys)
    add_gridlines_Z(Z[0])
    plt.savefig('{}/Z.pdf'.format(path_to_exp_results))
    plt.show()


    # Plot VP Trace

    # Plot mu vp mean
    trace_len = len(out['trace'])
    mu0_m_trace = torch.stack([-model.iota - t['mu0'].m.exp().cumsum(0)
                           for t in out['trace']])
    mu1_m_trace = torch.stack([model.iota + t['mu1'].m.exp().cumsum(0)
                           for t in out['trace']])

    plt.plot(mu0_m_trace.detach().numpy())
    plt.plot(mu1_m_trace.detach().numpy())
    plt.title('trace plot for $\mu$ vp mean')
    plt.show()

    # Plot W vp mean
    W_m_trace = torch.stack([model.sbt(t['W'].m) for t in out['trace']])
    for i in range(model.I):
        plt.plot(W_m_trace.detach().numpy()[:, i, :])
        if SIMULATE_DATA:
            for k in range(data['params']['W'].size(1)):
                plt.axhline(data['params']['W'][i, k])
        plt.title('trace plot for W_{} mean'.format(i))
        plt.show()


    # Plot sig vp mean
    sig_m_trace = torch.stack([t['sig'].m.exp() for t in out['trace']])
    plt.plot(sig_m_trace.detach().numpy())

    if SIMULATE_DATA:
        for i in range(model.I):
            plt.axhline(data['params']['sig'][i])

    plt.title('trace plot for $\sigma$ vp mean')
    plt.show()


    # Plot v vp mean
    v_m_trace = torch.stack([t['v'].m.sigmoid().cumprod(0) for t in out['trace']])
    plt.plot(v_m_trace.detach().numpy())
    plt.title('trace plot for v vp mean')
    plt.show()

