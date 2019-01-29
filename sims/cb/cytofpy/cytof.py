import torch
import math
import copy
import datetime
import numpy as np
import advi

from torch.distributions import Gamma
from torch.distributions import Beta
from torch.distributions import Normal
from torch.distributions import Dirichlet
from torch.distributions.log_normal import LogNormal
from torch.distributions import Uniform

class Cytof(advi.Model):
    def __init__(self, priors=None, dtype=torch.float64, device="cpu"):
        """
        TODO: Write doc
        """
        self.dtype = dtype
        self.device = device
        self.priors = priors

        self.I = None
        self.J = None
        self.N = None
        self.K = None
        self.L = None
        self.Nsum = None

    def __cache_model_constants__(self, data, K, L):
        self.K = K
        self.L = L
        self.I = len(data['y'])
        self.J = data['y'][0].size(1)
        self.N = [yi.size(0) for yi in data['y']]
        self.Nsum = sum(N)
        
        # Assert that all samples have equal number of markers (columns)
        for yi in data['y']:
            assert(yi.size(1) == self.J)
    
    def gen_default_priors(self, data, K=10, L=[5, 3],
                           sig_prior=log_normal(-2., 1.),
                           alpha_prior=Gamma(1., 1.),
                           mu_prior=None,
                           W_prior=None,
                           eta_prior=None,
                           eps_prior=Beta(5., 95.),
                           iota_prior=Gamma(1., 10.)):

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

        self.prior = {'mu': mu_prior, 'sig': sig_prior, 'eta': eta_prior,
                      'eta': eta_prior, 'eps': eps_prior, 'W': W_prior,
                      'iota': iota_prior, 'alpha': alpha_prior}

    def init_v(self): 
        """
        TODO
        """
        mu = [for z in range(2)]
        v = {'mu': ,
             'sig': None,
             'W': None,
             'v': None,
             'alpha': None,
             'eta': None,
             'eps': None,
             'iota': None}

        NotImplemented

    def subsample_data(self, data, minibatch_info=None):
        mini_data = []
        for i in range(self.I):
            n = int(minibatch_info['prop'] * self.N[i])
            idx = np.random.choice(N[i], n)
            mini_data.append()
        return mini_data

    def sample_real_params(self, v):
        NotImplemented


    def log_q(self, real_params, v):
        NotImplemented


    def loglike(self, real_params, data, minibatch_info=None):
        NotImplemented


    def to_real_space(self, params):
        NotImplemented

    def to_param_space(self, real_params):
        NotImplemented

    def msg(self, t, v):
        pass

