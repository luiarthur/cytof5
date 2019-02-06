import copy
import datetime
import math
import numpy as np
from Model import Model
from VarParams import *

import torch
from torch.distributions import Gamma
from torch.distributions import Beta
from torch.distributions import Normal
from torch.distributions import Dirichlet

class Cytof(Model):
    def __init__(self, data, K=None, L=None, iota=1.0, priors=None, dtype=torch.float64,
                 device="cpu", misc=None):
        super().__init__(data, priors, dtype, device, misc)

        # TODO:
        self.K = K
        self.L = L
        self.I =len(data['y'])
        self.N = [yi.size(0) for yi in data['y']]
        self.J = data['y'][0].size(1)
        self.Nsum = sum(self.N)
        self.debug = False
        self.iota = iota

        for i in range(self.I):
            assert data['y'][i].size(1) == self.J
            data['y'][i] = data['y'][i].reshape(self.N[i], self.J, 1, 1)

        if priors is None:
            self.gen_default_priors(K, L)
 
    def gen_default_priors(self, K, L,
                           sig_prior=Gamma(1, 1),
                           alpha_prior=Gamma(1., 1.),
                           mu0_prior=None,
                           mu1_prior=None,
                           W_prior=None,
                           eta0_prior=None,
                           eta1_prior=None):

        if L is None:
            L = [5, 3]
        
        self.L = L

        if K is None:
            K = 4

        self.K = K

        # FIXME: these should be an ordered prior of TruncatedNormals
        if mu0_prior is None:
            # mu0_prior = Uniform(-15, 0)
            mu0_prior = Gamma(1, 1)

        if mu1_prior is None:
            # mu1_prior = Uniform(0, 15)
            mu1_prior = Gamma(1, 1)

        if W_prior is None:
            W_prior = Dirichlet(torch.ones(self.K) / self.K)

        if eta0_prior is None:
            eta0_prior = Dirichlet(torch.ones(self.L[0]) / self.L[0])

        if eta1_prior is None:
            eta1_prior = Dirichlet(torch.ones(self.L[1]) / self.L[1])

        self.priors = {'mu0': mu0_prior, 'mu1': mu1_prior, 'sig': sig_prior,
                       'eta0': eta0_prior, 'eta1': eta1_prior,
                       'W': W_prior, 'alpha': alpha_prior}

    def log_q(self, params):
        out = 0.0
        for key in self.vp:
            out += self.vp[key].logpdf(params[key]).sum()
        if self.debug:
            print('log_q: {}'.format(out))
        return out / self.Nsum

    def loglike(self, data, params, minibatch_info):
        ll = 0.0
        
        for i in range(self.I):
            # Y: Ni x J x 1 x 1
            # muz: 1 x 1 x Lz x 1
            # etaz: I x J x Lz x 1
            # Ni x J x Lz x K
            d0 = Normal(-params['mu0'].cumsum(2), params['sig'][i]).log_prob(data['y'][i])
            d0 += params['eta0'][i:i+1, :, :, :].log()
            d1 = Normal(params['mu1'].cumsum(2), params['sig'][i]).log_prob(data['y'][i])
            d1 += params['eta1'][i:i+1, :, :, :].log()
            
            logmix_L0 = torch.logsumexp(d0, 2) # Ni x J x K
            logmix_L1 = torch.logsumexp(d1, 2) # Ni x J x K

            # Ni x J x K
            # Z: 1 x J x K
            b_vec = params['v'].cumprod(2)
            H = params['H']
            Z = ((b_vec - Normal(0,1).cdf(H)) / .5).sigmoid()
            c0 = Z * logmix_L1 + (1 - Z) * logmix_L0

            # OLD
            # Ni x K
            c = c0.sum(1)

            f = c + params['W'][i:i+1, :].log()
            lli = torch.logsumexp(f, 1).mean(0) * (self.N[i] / self.Nsum)
            assert(lli.dim() == 0)
            ll += lli

        if self.debug:
            print('log_like: {}'.format(ll))

        return ll

    def log_prior(self, params):
        lp_mu = 0.0
        for z in range(2):
            muz = 'mu0' if z == 0 else 'mu1'
            lp_mu += self.priors[muz].log_prob(params[muz].squeeze()).sum()

        lp_sig = self.priors['sig'].log_prob(params['sig'].squeeze()).sum()

        lp_v = Beta(params['alpha'],
                    torch.tensor(1.0)).log_prob(params['v'].squeeze()).sum()

        lp_alpha = Gamma(self.priors['alpha'].concentration,
                         self.priors['alpha'].rate).log_prob(params['alpha'])

        lp_W = 0.0
        for i in range(self.I):
            lp_W += self.priors['W'].log_prob(params['W'][i,:].squeeze())
    
        # v: 1 x 1 x K
        # Z: 1 x J x K
        lp_H = Normal(0, 1).log_prob(params['H']).sum()

        lp_eta = 0.0
        for z in range(2):
            etaz = 'eta0' if z == 0 else 'eta1'
            for i in range(self.I):
                for j in range(self.J):
                    tmp = self.priors[etaz].log_prob(params[etaz][i, j, :, 0].squeeze())
                    # print('i: {}, j:{}, lp_eta: {}'.format(i, j, tmp))
                    lp_eta += tmp

        lp = lp_mu + lp_sig + lp_v + lp_alpha + lp_W + lp_H + lp_eta 
        if self.debug:
            print('log_prior: {}'.format(lp))

        # return lp.sum() / self.Nsum
        return 0.0

    def msg(self, t):
        pass

    def subsample_data(self, minibatch_info=None):
        if minibatch_info is None:
            mini_data = self.data
        else:
            mini_data = {'y': [], 'm': []}
            for i in range(self.I):
                n = int(minibatch_info['prop'] * self.N[i])
                idx = np.random.choice(self.N[i], n)
                mini_data['y'].append(self.data['y'][i][idx, :, :, :])
                mini_data['m'].append(self.data['m'][i][idx, :])
        return mini_data

    def init_vp(self):
        self.vp = {'mu0': VPGamma((1, 1, self.L[0], 1)),
                   'mu1': VPGamma((1, 1, self.L[1], 1)),
                   'sig': VPGamma(self.I),
                   'W': VPDirichletW((self.I, self.K)),
                   'v': VPNormal((1, 1, self.K)),
                   'alpha': VPGamma(1),
                   'H': VPNormal((1, self.J, self.K)),
                   'eta0': VPDirichletEta((self.I, self.J, self.L[0], 1)),
                   'eta1': VPDirichletEta((self.I, self.J, self.L[1], 1))}
 
