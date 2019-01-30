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

def logit(p, a=0.0, b=1.0):
    return torch.log(p - a) - torch.log(b - p)

def lpdf_logitBeta(logitx, a, b):
    return trans.lpdf_logitx(logitx, Beta(a, b).logprob)

def lpdf_logitUniform(logitx, a, b):
    return trans.lpdf_logitx(logitx, Uniform(a, b).logprob, a, b)

def lpdf_logGamma(logx, shape, rate):
    return trans.lpdf_logx(logx, Gamma(shape, rate).logprob)

def lpdf_logLogNormal(logx, m, s):
    return trans.lpdf_logx(logx, LogNormal(m, s).logprob)

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
            self.priors = self.gen_default_priors(data=data, K=self.K, L=self.L)
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
        for yi in data['y']:
            assert(yi.size(1) == self.J)
    
    def gen_default_priors(self, data, K, L,
                           sig_prior=LogNormal(-2., 1.),
                           alpha_prior=Gamma(1., 1.),
                           mu_prior=None,
                           W_prior=None,
                           eta_prior=None,
                           eps_prior=Beta(5., 95.),
                           iota_prior=Gamma(1., 10.)):

        if L is None:
            L = [5, 3]

        self.__cache_model_constants__(data, K, L)

        if mu_prior is None:
            mu_prior = []
            # FIXME: this should be an ordered prior of TruncatedNormals
            mu_prior.append(Uniform(-15, 0))
            mu_prior.append(Uniform(0, 15))

        if W_prior is None:
            W_prior = Dirichlet(torch.ones(self.K))

        if eta_prior is None:
            eta_prior = []
            for z in range(2):
                eta_prior.append(Dirichlet(torch.ones(self.L[z])))

        self.priors = {'mu': mu_prior, 'sig': sig_prior, 'eta': eta_prior,
                       'eps': eps_prior, 'W': W_prior, 'iota': iota_prior,
                       'alpha': alpha_prior}

    def init_vp(self): 
        return {'mu': [VarParam(self.L[z]) for z in range(2)],
                'sig': VarParam(self.I),
                'W': [VarParam(self.K - 1) for i in range(self.I)],
                'v': VarParam(self.K),
                'alpha': VarParam(1),
                'eta': [VarParam((self.I, self.J, self.L[z] - 1)) for z in range(2)],
                'eps': VarParam(self.I),
                'iota': VarParam(1)}

    def subsample_data(self, data, minibatch_info=None):
        mini_data = []
        for i in range(self.I):
            n = int(minibatch_info['prop'] * self.N[i])
            idx = np.random.choice(self.N[i], n)
            mini_data.append(data['y'][i][idx, :, :, :])
        return mini_data

    def sample_real_params(self, vp):
        return {'mu': [muz.sample() for muz in vp['mu']],
                'sig': vp['sig'].sample(),
                'W': [wi.sample() for wi in vp['W']],
                'v': vp['v'].sample(),
                'alpha': vp['alpha'].sample(),
                'eta': [etaz.sample() for etaz in vp['eta']],
                'eps': vp['eps'].sample(),
                'iota': None}

    def log_q(self, real_params, vp):
        out = 0.0
        out += vp['mu'][0].log_prob(real_params['mu'][0])
        out += vp['mu'][1].log_prob(real_params['mu'][1])
        out += vp['sig'].log_prob(real_params['sig'])
        out += torch.stack([vp['W'][i].log_prob(real_params['W'][i])
                            for i in range(self.I)]).sum()
        out += vp['v'].log_prob(real_params['v'])
        out += vp['alpha'].log_prob(real_params['alpha'])
        out += torch.stack([vp['eta'][z].log_prob(real_params['eta'][z])
                            for z in range(2)]).sum()
        out += vp['eps'].log_prob(real_params['eps'])
        out += 0.0 # TODO for iota
        return out

    def log_prior(self, real_params):
        # FIXME. These should be ordered.
        lp_mu = 0.0
        for z in range(2):
            muz_min = self.priors['mu'][z].low
            muz_max = self.priors['mu'][z].high
            lp_mu += lpdf_logitUniform(real_params['mu'][z].squeeze(), muz_min, muz_max).sum()

        lp_sig = lpdf_logLogNormal(real_params['sig'].squeeze,
                                   self.priors['sig'].loc,
                                   self.priors['sig'].scale).sum()
        lp_W = 0.0
        for i in range(self.I):
            lp_W += lpdf_realDirichlet(real_params['W'][i],
                                       self.sbt_W,
                                       self.priors['W'][i].concentration)


        lp_v = lpdf_logitBeta(real_params['v'].squeeze(),
                              self.priors['v'].concentration0,
                              self.priors['v'].concentration1).sum()

        lp_alpha = lpdf_logGamma(real_params['alpha'],
                                 self.priors['alpha'].shape,
                                 self.priors['alpha'].rate)
        lp_eta = 0.0
        for z in range(2):
            for i in range(self.I):
                for j in range(self.J):
                    lp_eta += lpdf_realDirichlet(real_params['eta'][z][i, j, :],
                                                 self.sbt_eta[z],
                                                 self.priors['eta'][z].concentration)

        lp_eps = lpdf_logitBeta(real_params['eps'].squeeze(),
                                self.priors['eps'].concentration0,
                                self.priors['eps'].concentration1).sum()

        lp_iota = 0.0

        return lp_mu + lp_sig + lp_W + lp_v + lp_alpha + lp_eta + lp_eps + lp_iota

    def loglike(self, real_params, data, minibatch_info=None):
        params = self.to_param_space(real_params)
        ll = 0.0
        
        for i in range(self.I):
            d0 = Normal(params['mu'][0], params['sig'][i]).log_prob(data['y'][i])
            d0 += torch.log(params['eta'][0][i])
            d1 = Normal(params['mu'][1], params['sig'][i]).log_prob(data['y'][i])
            d1 += torch.log(params['eta'][1][i])

            a0 = torch.logsumexp(d1, 2) # Ni x J x K
            a1 = torch.logsumexp(d0, 2) # Ni x J x K

            c = (a1 * torch.log(params['v']) + a0 * torch.log(1-params['v'])).sum(1) # Ni x K
            f = c + torch.log(params['W'][i])
            lli = torch.logsumexp(f, 1).sum()

            if minibatch_info is None:
                ll += lli
            else:
                ll += data.N[i] * lli / minibatch_info['n']

        return ll

    def to_real_space(self, params):
        eta0 = torch.empty(self.I, self.J, self.L[0] - 1)
        eta1 = torch.empty(self.I, self.J, self.L[1] - 1)
        for i in range(self.I):
            for j in range(self.J):
                eta0[i, j, :] = self.sbt_eta[0].inv(params['eta'][0][i, j, :])
                eta1[i, j, :] = self.sbt_eta[1].inv(params['eta'][1][i, j, :])

        return {'mu': params['mu'], # FIXME. This should be truncated ideally.
                'sig': torch.log(params['sig']),
                'W': [self.sbt_W.inv(params['W'][i]) for i in range(self.I)],
                'v': logit(params['v']),
                'alpha': torch.log(params['alpha']),
                'eta': [eta0, eta1],
                'eps': torch.logit(params['eps']),
                'iota': None}

    def to_param_space(self, real_params):
        eta0 = torch.empty(self.I, self.J, self.L[0])
        eta1 = torch.empty(self.I, self.J, self.L[1])
        for i in range(self.I):
            for j in range(self.J):
                eta0[i, j, :] = self.sbt_eta[0](real_params['eta'][0][i, j, :])
                eta1[i, j, :] = self.sbt_eta[1](real_params['eta'][1][i, j, :])

        return {'mu': real_params['mu'], # FIXME. This should be truncated ideally.
                'sig': torch.exp(real_params['sig']),
                'W': self.sbt_W(real_params['W']),
                'v': torch.sigmoid(real_params['v']),
                'alpha': torch.exp(real_params['alpha']),
                'eta': [eta0, eta1],
                'eps': torch.sigmoid(real_params['eps']),
                'iota': None}

    def msg(self, t, vp):
        pass

