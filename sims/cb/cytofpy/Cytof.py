import copy
import datetime
import math
import numpy as np

import torch
from torch.distributions import Gamma
from torch.distributions import Beta
from torch.distributions import Normal
from torch.distributions import Dirichlet
from torch.distributions.log_normal import LogNormal
from torch.distributions import Uniform
from torch.distributions.transforms import StickBreakingTransform

import advi
import advi.transformations as trans
from VarParam import VarParam


def lpdf_logitBeta(logitx, a, b):
    return trans.lpdf_logitx(logitx, Beta(a, b).log_prob)

def lpdf_logitUniform(logitx, a, b):
    return trans.lpdf_logitx(logitx, Uniform(a, b).log_prob, a, b)

def lpdf_logGamma(logx, shape, rate):
    return trans.lpdf_logx(logx, Gamma(shape, rate).log_prob)

def lpdf_logLogNormal(logx, m, s):
    return trans.lpdf_logx(logx, LogNormal(m, s).log_prob)

def lpdf_realDirichlet(real_x, sbt, a):
    """
    real_x should be squeezed
    """
    lpdf = Dirichlet(a).log_prob
    simplex_x = sbt(real_x)
    return lpdf(simplex_x) + sbt.log_abs_det_jacobian(real_x, simplex_x)


class Cytof(advi.Model):
    def __init__(self, data, priors=None, K=None, L=None, dtype=torch.float64, device="cpu"):
        """
        TODO: Write doc
        """
        self.dtype = dtype
        self.device = device

        self.I = None
        self.J = None
        self.N = None
        self.debug = False

        if K is None:
            self.K = 10
        else:
            self.K = K

        if L is None:
            self.L = [5, 3]
        else:
            self.L = L

        self.Nsum = None
        self.sbt_W = StickBreakingTransform(1)
        self.sbt_eta = [StickBreakingTransform(1), StickBreakingTransform(1)]

        if priors is None:
            self.gen_default_priors(data=data, K=self.K, L=self.L)
        else:
            self.priors = priors

    def __cache_model_constants__(self, data, K, L):
        self.K = K
        self.L = L
        self.I = len(data['y'])
        self.J = data['y'][0].size(1)
        self.N = [yi.size(0) for yi in data['y']]
        self.Nsum = sum(self.N)

        # Assert that all samples have equal number of markers (columns)
        for i in range(self.I):
            assert(data['y'][i].size(1) == self.J)
            data['y'][i] = data['y'][i].reshape(self.N[i], self.J, 1, 1)
    
    def gen_default_priors(self, data, K, L,
                           sig_prior=LogNormal(-2., 1.),
                           alpha_prior=Gamma(1., 1.),
                           mu0_prior=None,
                           mu1_prior=None,
                           W_prior=None,
                           eta0_prior=None,
                           eta1_prior=None,
                           eps_prior=Beta(5., 95.),
                           iota_prior=Gamma(1., 10.)):

        if L is None:
            L = [5, 3]

        self.__cache_model_constants__(data, K, L)

        # FIXME: these should be an ordered prior of TruncatedNormals
        if mu0_prior is None:
            mu0_prior = Uniform(-15, 0)

        if mu1_prior is None:
            mu1_prior = Uniform(0, 15)

        if W_prior is None:
            W_prior = Dirichlet(torch.ones(self.K) / self.K)

        if eta0_prior is None:
            eta0_prior = Dirichlet(torch.ones(self.L[0]))

        if eta1_prior is None:
            eta1_prior = Dirichlet(torch.ones(self.L[1]))

        self.priors = {'mu0': mu0_prior, 'mu1': mu1_prior, 'sig': sig_prior,
                       'eta0': eta0_prior, 'eta1': eta1_prior,
                       'eps': eps_prior, 'W': W_prior, 'iota': iota_prior,
                       'alpha': alpha_prior}

    def init_vp(self): 
        return {'mu0': VarParam((1, 1, self.L[0], 1)),
                'mu1': VarParam((1, 1, self.L[1], 1)),
                'sig': VarParam(self.I),
                'W': VarParam((self.I, self.K - 1), init_m=0.0, init_log_s=-1.0),
                'v': VarParam((1, 1, self.K)),
                'alpha': VarParam(1),
                'eta0': VarParam((self.I, self.J, self.L[0] - 1, 1),
                                 init_m=0.0, init_log_s=-1.0),
                'eta1': VarParam((self.I, self.J, self.L[1] - 1, 1),
                                 init_m=0.0, init_log_s=-1.0),
                'eps': VarParam(self.I),
                'iota': VarParam(1)}

    def subsample_data(self, data, minibatch_info=None):
        if minibatch_info is None:
            mini_data = data
        else:
            mini_data = {'y': [], 'm': []}
            for i in range(self.I):
                n = int(minibatch_info['prop'] * self.N[i])
                idx = np.random.choice(self.N[i], n)
                mini_data['y'].append(data['y'][i][idx, :, :, :])
                mini_data['m'].append(data['m'][i][idx, :])
        return mini_data

    def sample_real_params(self, vp):
        real_params = {'iota': None}
        for key in vp:
            if key != 'iota':
                real_params[key] = vp[key].sample()
        return real_params

    def log_q(self, real_params, vp):
        out = 0.0
        for key in vp:
            if key != 'iota' and key != 'eps':
                out += vp[key].log_prob(real_params[key]).sum()
        if self.debug:
            print('log_q: {}'.format(out))
        return out / self.Nsum

    def log_prior(self, real_params):
        # FIXME. These should be ordered.
        lp_mu = 0.0
        for z in range(2):
            muz = 'mu0' if z == 0 else 'mu1'
            muz_min = self.priors[muz].low
            muz_max = self.priors[muz].high
            lp_mu += lpdf_logitUniform(real_params[muz].squeeze(), muz_min, muz_max).sum()

        lp_sig = lpdf_logLogNormal(real_params['sig'].squeeze(),
                                   self.priors['sig'].loc,
                                   self.priors['sig'].scale).sum()
        lp_W = 0.0
        for i in range(self.I):
            lp_W += lpdf_realDirichlet(real_params['W'][i, :],
                                       self.sbt_W,
                                       self.priors['W'].concentration)

        lp_v = lpdf_logitBeta(real_params['v'].squeeze(),
                              torch.exp(real_params['alpha']) / self.K,
                              torch.tensor(1.0)).sum()

        lp_alpha = lpdf_logGamma(real_params['alpha'],
                                 self.priors['alpha'].concentration,
                                 self.priors['alpha'].rate).sum()
        lp_eta = 0.0
        for z in range(2):
            etaz = 'eta0' if z == 0 else 'eta1'
            for i in range(self.I):
                for j in range(self.J):
                    tmp = lpdf_realDirichlet(real_params[etaz][i, j, :, 0].squeeze(),
                                             self.sbt_eta[z],
                                             self.priors[etaz].concentration)
                    # print('i: {}, j:{}, lp_eta: {}'.format(i, j, tmp))
                    lp_eta += tmp
                    # lp_eta += 0.0 # FIXME. nan

        # lp_eps = lpdf_logitBeta(real_params['eps'].squeeze(),
        #                         self.priors['eps'].concentration0,
        #                         self.priors['eps'].concentration1).sum()
        lp_eps = 0.0

        lp_iota = 0.0

        lp = lp_mu + lp_sig + lp_W + lp_v + lp_alpha + lp_eta + lp_eps + lp_iota
        if self.debug:
            print('log_prior:       {}'.format(lp))
            print('log_prior mu:    {}'.format(lp_mu))
            print('log_prior sig:   {}'.format(lp_sig))
            print('log_prior W:     {}'.format(lp_W))
            print('log_prior v:     {}'.format(lp_v))
            print('log_prior alpha: {}'.format(lp_alpha))
            print('log_prior eta:   {}'.format(lp_eta))
            print('log_prior eps:   {}'.format(lp_eps))
        return lp / self.Nsum

    def loglike(self, real_params, data, minibatch_info=None):
        params = self.to_param_space(real_params)
        ll = 0.0
        
        # FIXME: Check this!
        for i in range(self.I):
            # Ni x J x Lz x K
            d0 = Normal(params['mu0'], params['sig'][i]).log_prob(data['y'][i])
            d0 += torch.log(params['eta0'][i:i+1, :, :, :])
            d1 = Normal(params['mu1'], params['sig'][i]).log_prob(data['y'][i])
            d1 += torch.log(params['eta1'][i:i+1, :, :, :])
            
            a0 = torch.logsumexp(d0, 2) # Ni x J x K
            a1 = torch.logsumexp(d1, 2) # Ni x J x K

            # Ni x J x K
            c0 = torch.logsumexp(torch.stack([
                  a1 + torch.log(params['v']),
                  a0 + torch.log1p(-params['v'])]), 0)
            # Ni x K
            c = c0.sum(1)

            f = c + torch.log(params['W'][i:i+1, :])
            # lli = torch.logsumexp(f, 1).sum()
            lli = torch.logsumexp(f, 1).mean() * (self.N[i] / self.Nsum)

            ll += lli

            # print(lli)

            # if minibatch_info is None:
            #     ll += lli
            # else:
            #     n = int(minibatch_info['prop'] * self.N[i])
            #     ll += self.N[i] * lli / n

        if self.debug:
            print('log_like: {}'.format(ll))
        return ll

    def to_real_space(self, params):
        eta0 = torch.empty(self.I, self.J, self.L[0] - 1, 1)
        eta1 = torch.empty(self.I, self.J, self.L[1] - 1, 1)
        for i in range(self.I):
            for j in range(self.J):
                eta0[i, j, :, 0] = self.sbt_eta[0].inv(params['eta0'][i, j, :, 0])
                eta1[i, j, :, 0] = self.sbt_eta[1].inv(params['eta1'][i, j, :, 0])

        mu0 = trans.logit(params['mu0'], self.priors['mu0'].low, self.priors['mu0'].high)
        mu1 = trans.logit(params['mu1'], self.priors['mu1'].low, self.priors['mu1'].high)
        return {# FIXME. This should be truncated ideally.
                'mu0': mu0,
                'mu1': mu1,
                #
                'sig': torch.log(params['sig']),
                'W': torch.stack([self.sbt_W.inv(params['W'][i, :])
                                  for i in range(self.I)]),
                'v': logit(params['v']),
                'alpha': torch.log(params['alpha']),
                'eta0': eta0,
                'eta1': eta1,
                'eps': torch.logit(params['eps']),
                'iota': None}

    def to_param_space(self, real_params):
        eta0 = torch.empty(self.I, self.J, self.L[0], 1)
        eta1 = torch.empty(self.I, self.J, self.L[1], 1)
        for i in range(self.I):
            for j in range(self.J):
                eta0[i, j, :, 0] = self.sbt_eta[0](real_params['eta0'][i, j, :, 0])
                eta1[i, j, :, 0] = self.sbt_eta[1](real_params['eta1'][i, j, :, 0])

        mu0 = trans.invlogit(real_params['mu0'], self.priors['mu0'].low,
                             self.priors['mu0'].high)
        mu1 = trans.invlogit(real_params['mu1'], self.priors['mu1'].low,
                             self.priors['mu1'].high)
        return {# FIXME. This should be truncated ideally.
                'mu0': mu0,
                'mu1': mu1,
                #
                'sig': torch.exp(real_params['sig']),
                'W': self.sbt_W(real_params['W']),
                'v': torch.sigmoid(real_params['v']),
                'alpha': torch.exp(real_params['alpha']),
                'eta0': eta0,
                'eta1': eta1,
                'eps': torch.sigmoid(real_params['eps']),
                'iota': None}

    def msg(self, t, vp):
        pass
        # for key in vp:
        #     print('{}: {}'.format(key, vp[key].m))


    def fit(self, data, niters:int=1000, nmc:int=2, lr:float=1e-2,
            minibatch_info=None, seed:int=1, eps:float=1e-6, init=None,
            print_freq:int=10, verbose:int=1):
        """
        fir the model.

        data: data
        niter: max number of iterations for ADVI
        nmc: number of MC samples for estimating ELBO mean (default=2). nmc=1
             is usually sufficient. nmc >= 2 may be required for noisy gradients.
             nmc >= 10 is overkill in most situations.
        lr: learning rate (> 0)
        minibatch_info: information on minibatches
        seed: random seed for torch (for reproducibility)
        eps: threshold for determining convergence. If `abs((elbo_curr /
             elbo_prev) -1) < eps`, then ADVI exits before `niter` iterations.
        init: initial values for variational parameters (in real space). This has
              the same for as the output.
        print_freq: how often to print ELBO value during algorithm. For monitoring
                    status of ADVI. (default=10, i.e. print every 10 iterations.)
        verbose: an integer indicating how much output to show. defaults to 1, 
                 which prints the ELBO. Setting verbose=0 will turn off all outputs.

        returns: dictionary with keys 'v' and 'elbo', where 'v' is the
                 variational parameters in real space, and 'elbo' is the 
                 ELBO history.
        """

        assert(nmc >= 1)
        assert(lr > 0)
        assert(eps >= 0)


        if init is not None:
            vp = copy.deepcopy(init)
        else:
            vp = self.init_vp()

        optimizer = torch.optim.Adam([vpj.m for vpj in vp.values()] + 
                                     [vpj.log_s for vpj in vp.values()], lr=lr)
        elbo = []

        for t in range(niters):
            elbo_mean = self.compute_elbo_mean(data, vp, nmc, minibatch_info)
            loss = -elbo_mean
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            elbo.append(elbo_mean.item())

            if print_freq > 0 and (t + 1) % print_freq == 0:
                now = datetime.datetime.now().replace(microsecond=0)
                if verbose >= 1:
                    print('{} | iteration: {}/{} | elbo mean: {}'.format(
                        now, t + 1, niters, elbo[-1]))
                    
                if verbose >= 2:
                    print('state: {}'.format(vp))

                self.msg(t, vp)

            if t > 0 and abs(elbo[-1] / elbo[-2] - 1) < eps:
                print("Convergence suspected. Ending optimizer early.")
                break

            if math.isnan(elbo[-1]):
                print("nan detected! Exiting optimizer early.")
                break

        return {'vp': vp, 'elbo': elbo}

