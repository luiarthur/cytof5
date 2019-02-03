import copy
import datetime
import math
import numpy as np

import torch
from torch.distributions import Gamma
from torch.distributions import Bernoulli
from torch.distributions import Beta
from torch.distributions import Normal
from torch.distributions import Dirichlet
from torch.distributions.log_normal import LogNormal
from torch.distributions import Uniform
from torch.distributions.transforms import StickBreakingTransform

import advi
import advi.transformations as trans
from VarParam import VarParam, VarParamBernoulli


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
            self.__cache_model_constants__(data, K, L)
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
                           # sig_prior=LogNormal(-2., 1.),
                           sig_prior=Gamma(100, 1000),
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
            # mu0_prior = Uniform(-15, 0)
            mu0_prior = Gamma(1, 1)

        if mu1_prior is None:
            # mu1_prior = Uniform(0, 15)
            mu1_prior = Gamma(1, 1)

        if W_prior is None:
            W_prior = Dirichlet(torch.ones(self.K) / self.K)

        if eta0_prior is None:
            eta0_prior = Dirichlet(torch.ones(self.L[0])) # / self.L[0])

        if eta1_prior is None:
            eta1_prior = Dirichlet(torch.ones(self.L[1])) #/ self.L[1])

        self.priors = {'mu0': mu0_prior, 'mu1': mu1_prior, 'sig': sig_prior,
                       'eta0': eta0_prior, 'eta1': eta1_prior,
                       'eps': eps_prior, 'W': W_prior, 'iota': iota_prior,
                       'alpha': alpha_prior}

    def init_vp(self): 
        return {'mu0': VarParam((1, 1, self.L[0], 1), init_m=1, init_log_s=0),
                'mu1': VarParam((1, 1, self.L[1], 1), init_m=1, init_log_s=0),
                'sig': VarParam(self.I, init_m=-1, init_log_s=0),
                'W': VarParam((self.I, self.K - 1)),
                'v': VarParam((1, 1, self.K)),
                'alpha': VarParam(1),
                'Z': VarParamBernoulli((1, self.J, self.K)),
                'eta0': VarParam((self.I, self.J, self.L[0] - 1, 1)),
                'eta1': VarParam((self.I, self.J, self.L[1] - 1, 1)),
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
        real_params = {}
        for key in vp:
            if key != 'iota' and key != 'eps':
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
            lp_mu += lpdf_logGamma(real_params[muz].squeeze(),
                                   self.priors[muz].concentration,
                                   self.priors[muz].rate).sum()

        lp_sig = lpdf_logGamma(real_params['sig'].squeeze(),
                               self.priors['sig'].concentration,
                               self.priors['sig'].rate).sum()

        lp_W = 0.0
        for i in range(self.I):
            lp_W += lpdf_realDirichlet(real_params['W'][i, :].squeeze(),
                                       self.sbt_W,
                                       self.priors['W'].concentration)

        lp_v = lpdf_logitBeta(real_params['v'].squeeze(),
                              torch.exp(real_params['alpha']),
                              torch.tensor(1.0)).sum()

        lp_alpha = lpdf_logGamma(real_params['alpha'],
                                 self.priors['alpha'].concentration,
                                 self.priors['alpha'].rate).sum()

        # Actually, Z is binary here. This is correct, but easier to code.
        # v: 1 x 1 x K
        # Z: 1 x J x K
        # lp_Z = Bernoulli(torch.sigmoid(real_params['v'])).log_prob(real_params['Z']).sum()
        b_vec = torch.sigmoid(real_params['v']).cumprod(2)
        lp_Z = (real_params['Z'] * b_vec.log() + (1 - real_params['Z']) * (1 - b_vec).log()).sum()

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

        # lp_eps = lpdf_logitBeta(real_params['eps'].squeeze(),
        #                         self.priors['eps'].concentration0,
        #                         self.priors['eps'].concentration1).sum()
        lp_eps = 0.0

        lp_iota = 0.0

        lp = lp_mu + lp_sig + lp_W + lp_v + lp_alpha + lp_eta + lp_eps + lp_iota + lp_Z
        # lp = lp_mu + lp_sig + lp_v + lp_alpha + lp_Z
        if self.debug:
            print('log_prior:       {}'.format(lp))
            print('log_prior mu:    {}'.format(lp_mu))
            print('log_prior sig:   {}'.format(lp_sig))
            print('log_prior W:     {}'.format(lp_W))
            print('log_prior v:     {}'.format(lp_v))
            print('log_prior Z:     {}'.format(lp_Z))
            print('log_prior alpha: {}'.format(lp_alpha))
            print('log_prior eta:   {}'.format(lp_eta))
            print('log_prior eps:   {}'.format(lp_eps))

        # TODO: use the priors!
        return lp / self.Nsum
        # return 0.0

    def loglike(self, real_params, data, minibatch_info=None):
        params = self.to_param_space(real_params)
        if self.debug:
            print(params)

        ll = 0.0
        
        # FIXME: Check this!
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
            c0 = params['Z'] * logmix_L1 + (1 - params['Z']) * logmix_L0

            # OLD
            # Ni x K
            c = c0.sum(1)

            f = c + params['W'][i:i+1, :].log()
            lli = torch.logsumexp(f, 1).mean(0) * (self.N[i] / self.Nsum)

            # NEW
            # log_W = params['W'][i, :].reshape(1, 1, self.K).log()
            # lli = (log_W + c0).logsumexp(2).sum() * (self.N[i] / self.Nsum)


            ll += lli

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

        # mu0 = trans.logit(params['mu0'], self.priors['mu0'].low, self.priors['mu0'].high)
        # mu1 = trans.logit(params['mu1'], self.priors['mu1'].low, self.priors['mu1'].high)
        mu0 = torch.log(params['mu0'])
        mu1 = torch.log(params['mu1'])
        return {'mu0': mu0,
                'mu1': mu1,
                'sig': torch.log(params['sig']),
                'W': torch.stack([self.sbt_W.inv(params['W'][i, :])
                                  for i in range(self.I)]),
                'v': logit(params['v']),
                'Z': params['Z'],
                'alpha': torch.log(params['alpha']),
                'eta0': eta0,
                'eta1': eta1,
                'eps': None, #torch.logit(params['eps']),
                'iota': None}

    def to_param_space(self, real_params):
        eta0 = torch.empty(self.I, self.J, self.L[0], 1)
        eta1 = torch.empty(self.I, self.J, self.L[1], 1)
        W = torch.empty(self.I, self.K)

        for i in range(self.I):
            W[i, :] = self.sbt_W(real_params['W'][i, :].squeeze())
            for j in range(self.J):
                eta0[i, j, :, 0] = self.sbt_eta[0](real_params['eta0'][i, j, :, 0])
                eta1[i, j, :, 0] = self.sbt_eta[1](real_params['eta1'][i, j, :, 0])

        mu0 = torch.exp(real_params['mu0'])
        mu1 = torch.exp(real_params['mu1'])
        return {'mu0': mu0,
                'mu1': mu1,
                'sig': torch.exp(real_params['sig']),
                # 'W': self.sbt_W(real_params['W']),
                'W': W,
                'v': torch.sigmoid(real_params['v']),
                'Z': real_params['Z'],
                'alpha': torch.exp(real_params['alpha']),
                'eta0': eta0,
                'eta1': eta1,
                'eps': None, #torch.sigmoid(real_params['eps']),
                'iota': None}

    def msg(self, t, vp):
        pass
        # for key in vp:
        #     print('{}: {}'.format(key, vp[key].m))


    def fit(self, data, niters:int=1000, nmc:int=2, lr:float=1e-2,
            minibatch_info=None, seed:int=1, eps:float=1e-6, init=None,
            print_freq:int=10, verbose:int=1, trace_vp:bool=False):
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
        trace_vp: Boolean. Whether or not to store the variational parameters.
                  Mostly for testing. Don't store if storage and memory are issues.

        returns: dictionary with keys 'v' and 'elbo', where 'v' is the
                 variational parameters in real space, and 'elbo' is the 
                 ELBO history.
        """
        torch.manual_seed(seed)

        assert(nmc >= 1)
        assert(lr > 0)
        assert(eps >= 0)
        

        if init is not None:
            vp = copy.deepcopy(init)
        else:
            vp = self.init_vp()
        
        param_names = vp.keys()
        param_names_except_Z = list(filter(lambda x: x != 'Z', param_names))

        optimizer = torch.optim.Adam([vp[key].m for key in param_names_except_Z] + 
                                     [vp[key].log_s for key in param_names_except_Z] + 
                                     [vp['Z'].logit_p], lr=lr)
        elbo = []
        
        best_vp = copy.deepcopy(vp)
        trace = []

        for t in range(niters):
            elbo_mean = self.compute_elbo_mean(data, vp, nmc, minibatch_info)
            loss = -elbo_mean
            optimizer.zero_grad()
            loss.backward(retain_graph=True)

            fixed_grad = False
            with torch.no_grad():
                for key in vp:
                    if key != 'iota' and key != 'eps' and key != 'Z':
                        grad_m_isnan = torch.isnan(vp[key].m.grad)
                        if grad_m_isnan.sum() > 0:
                            print("WARNING: Setting a nan gradient to zero in {}!".format(key))
                            vp[key].m.grad[grad_m_isnan] = 0.0
                            fixed_grad = True

                        grad_log_s_isnan = torch.isnan(vp[key].log_s.grad)
                        if grad_log_s_isnan.sum() > 0:
                            print("WARNING: Setting a nan gradient to zero in {}!".format(key))
                            vp[key].log_s.grad[grad_log_s_isnan] = 0.0
                            fixed_grad = True

                    elif key == 'Z':
                        grad_logit_p_isnan = torch.isnan(vp[key].logit_p.grad)
                        if grad_logit_p_isnan.sum() > 0:
                            print("WARNING: Setting a nan gradient to zero in {}!".format(key))
                            vp[key].logit_p.grad[grad_logit_p_isnan] = 0.0
                            fixed_grad = True

            if fixed_grad:
                for key in vp:
                    if key != 'iota' and key != 'eps':
                        with torch.no_grad():
                            if key != 'Z':
                                vp[key].m.data = best_vp[key].m.data
                                vp[key].log_s.data = best_vp[key].log_s.data
                            else:
                                vp[key].logit_p.data = best_vp[key].logit_p.data

            if t % 10 == 0 and not fixed_grad:
                # TODO: Save this periodically
                best_vp = copy.deepcopy(vp)

                # Trace the vp
                trace.append(copy.deepcopy(vp))

            optimizer.step()
            elbo.append(elbo_mean.item())

            # if fixed_grad:
            #     print('Throwing elbo from history because of nan in gradients.')
            # else:

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
                if t > 20 and math.isnan(elbo[-2]):
                    print("ELBO is becoming nan. Terminating optimizer early.")
                    vp = best_vp
                    break

        return {'vp': vp, 'elbo': elbo, 'trace': trace}

