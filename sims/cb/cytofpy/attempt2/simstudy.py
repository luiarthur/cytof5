import os
import torch

from Cytof import Cytof

import sys
sys.path.append('../attempt1/')

from readCB import readCB
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

    torch.manual_seed(2)
    np.random.seed(0)

    cm_greys = plt.cm.get_cmap('Greys')
    
    # data = simdata(N=[30000, 10000, 20000], L0=3, L1=3, J=12, K=4)
    # data = simdata(N=[3000, 3000, 3000], L0=3, L1=3, J=12, K=4)
    data = simdata(N=[3000, 1000, 2000], L0=3, L1=3, J=6, a_W=[300, 200, 500])
    cb = data['data']
    plt.imshow(data['params']['Z'], aspect='auto', vmin=0, vmax=1, cmap=cm_greys)
    J, K = data['params']['Z'].shape
    add_gridlines_Z(data['params']['Z'])
    plt.savefig('{}/Z_true.pdf'.format(path_to_exp_results))
    plt.show()

    y = copy.deepcopy(cb['y'])

    plt.hist(y[0][:, 1], bins=100, density=True); plt.xlim(-7, 7); plt.show()
    plt.hist(y[1][:, 3], bins=100, density=True); plt.xlim(-7, 7); plt.show()
    plt.hist(y[2][:, -1], bins=100, density=True); plt.xlim(-7, 7); plt.show()

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

    K = 3
    model = Cytof(data=cb, K=K, L=[3,3])
    model.debug=True
    out = model.fit(niters=1000, lr=1e-3, print_freq=10, eps=1e-6,
                    # minibatch_info={'prop': .9},
                    nmc=1)

    # Save output
    pickle.dump(out, open('{}/out.p'.format(path_to_exp_results), 'wb'))

    elbo = out['elbo']
    vp = out['vp']

    out = pickle.load(open('{}/out.p'.format(path_to_exp_results), 'rb'))

    plt.plot(elbo)
    plt.ylabel('ELBO / NSUM')
    plt.show()

    # Posterior Inference
    B = 100
    post = [model.sample_params() for b in range(B)]

    # Plot Z
    Z = torch.stack([p['Z'] for p in post]).detach().reshape((B, model.J, model.K)).numpy()
    plt.imshow(Z.mean(0) > .5, aspect='auto', vmin=0, vmax=1, cmap=cm_greys)
    add_gridlines_Z(Z.mean(0))
    plt.savefig('{}/Z.pdf'.format(path_to_exp_results))
    plt.show()


