import copy
import datetime
import math
import numpy as np
from Model import Model
from VarDist import *

import torch
from torch.distributions import Gamma
from torch.distributions import Beta
from torch.distributions import Normal
from torch.distributions import Dirichlet
from torch.distributions.kl import kl_divergence

class Cytof(Model):
    def __init__(self, data, K=None, L=None, iota=1.0, priors=None, tau=0.5,
                 dtype=torch.float64, device="cpu", misc=None):
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
        self.tau = tau

        for i in range(self.I):
            assert data['y'][i].size(1) == self.J
            data['y'][i] = data['y'][i].reshape(self.N[i], self.J)

        if priors is None:
            self.gen_default_priors(K, L)
 
    def gen_default_priors(self, K, L,
                           sig_prior=LogNormal(0, 1),
                           alpha_prior=Gamma(1., 10.),
                           mu0_prior=None,
                           mu1_prior=None,
                           W_prior=None,
                           eta0_prior=None,
                           eta1_prior=None):

        if L is None:
            L = [5, 5]
        
        self.L = L

        if K is None:
            K = 30

        self.K = K

        # FIXME: these should be an ordered prior of TruncatedNormals
        if mu0_prior is None:
            mu0_prior = Gamma(1, 1)

        if mu1_prior is None:
            mu1_prior = Gamma(1, 1)

        if W_prior is None:
            W_prior = Dirichlet(torch.ones(self.K) / self.K)

        if eta0_prior is None:
            eta0_prior = Dirichlet(torch.ones(self.L[0]) / self.L[0])

        if eta1_prior is None:
            eta1_prior = Dirichlet(torch.ones(self.L[1]) / self.L[1])

        self.priors = {'mu0': mu0_prior, 'mu1': mu1_prior, 'sig': sig_prior,
                       'H': Normal(0, 1),
                       'eta0': eta0_prior, 'eta1': eta1_prior,
                       'W': W_prior, 'alpha': alpha_prior}

    def kl_qp(self, params):
        res = 0.0

        for key in self.vp:
            if key != 'v':
                res += kl_divergence(self.vp[key].dist(), self.priors[key]).sum()

        # v
        res += kl_divergence(self.vp['v'].dist(), Beta(params['alpha'], 1)).sum()

        return res / self.Nsum
 
    def loglike(self, data, params, minibatch_info):
        ll = 0.0
        
        # FIXME: Check this!
        for i in range(self.I):
            # Y: Ni x J
            # muz: Lz
            # etaz_i: 1 x J x Lz

            # Ni x J x Lz
            d0 = Normal(-self.iota - params['mu0'].cumsum(0)[None, None, :],
                        params['sig'][i]).log_prob(data['y'][i][:, :, None])
            d0 += params['eta0'][i:i+1, :, :].log()

            d1 = Normal(self.iota + params['mu1'].cumsum(0)[None, None, :],
                        params['sig'][i]).log_prob(data['y'][i][:, :, None])
            d1 += params['eta1'][i:i+1, :, :].log()
            
            # Ni x J
            logmix_L0 = torch.logsumexp(d0, 2)
            logmix_L1 = torch.logsumexp(d1, 2)

            # Z: J x K
            # H: J x K
            # v: K
            # c: Ni x J x K
            # d: Ni x K
            # Ni x J x K

            # FIXME: USING A SIGMOID HERE TOTALLY HELPS!!!
            #        IS IT HACKY? FIND SOMETHING STEEPER THAN SIGMOID
            b_vec = params['v'].cumprod(0)
            Z = ((b_vec[None, :] - Normal(0, 1).cdf(params['H'])) / self.tau).sigmoid()
            c = Z[None, :] * logmix_L1[:, :, None] + (1 - Z[None, :]) * logmix_L0[:, :, None]
            d = c.sum(1)

            f = d + params['W'][i:i+1, :].log()
            lli = torch.logsumexp(f, 1).mean(0) * (self.N[i] / self.Nsum)
            assert(lli.dim() == 0)

            ll += lli

        if self.debug:
            print('log_like: {}'.format(ll))
        return ll


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
                mini_data['y'].append(self.data['y'][i][idx, :])
                mini_data['m'].append(self.data['m'][i][idx, :])
        return mini_data

    def init_vp(self):
        self.vp = {'mu0': VDGamma(self.L[0]),
                   'mu1': VDGamma(self.L[1]),
                   'sig': VDLogNormal(self.I),
                   'W': VDDirichlet((self.I, self.K)),
                   'v': VDBeta(self.K),
                   'alpha': VDGamma(1),
                   'H': VDNormal((self.J, self.K)),
                   'eta0': VDDirichlet((self.I, self.J, self.L[0])),
                   'eta1': VDDirichlet((self.I, self.J, self.L[1]))}
 
