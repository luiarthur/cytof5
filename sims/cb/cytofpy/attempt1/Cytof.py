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


def lpdf_logitBeta(logitx, prior):
    return trans.lpdf_logitx(logitx.clamp(-6, 6), prior.log_prob)

def lpdf_logitUniform(logitx, prior):
    return trans.lpdf_logitx(logitx.clamp(-6, 6), prior.log_prob, prior.low, prior.high)

def lpdf_logx(logx, prior):
    return trans.lpdf_logx(logx, prior.log_prob)

def lpdf_realDirichlet(real_x, prior):
    """
    real_x should be squeezed
    """
    real_x.clamp_(-6, 6)
    lpdf = prior.log_prob
    sbt = StickBreakingTransform(0)
    simplex_x = sbt(real_x) + .01
    return lpdf(simplex_x) + sbt.log_abs_det_jacobian(real_x, simplex_x)
    


class Cytof(advi.Model):
    def __init__(self, data, priors=None, K=None, L=None, iota=1.0, tau=0.5,
                 dtype=torch.float64, device="cpu"):
        """
        TODO: Write doc
        tau: temperature parameter for sigmoid gumbel for Z.
             tau should be in the unit interval.
             Smaller tau makes Z_jk closer to extremes (0, 1).
        """
        self.dtype = dtype
        self.device = device

        assert(0.0 < tau < 1.0)

        self.I = None
        self.J = None
        self.N = None
        self.debug = False
        self.iota = iota
        self.tau = tau

        if K is None:
            self.K = 10
        else:
            self.K = K

        if L is None:
            self.L = [5, 3]
        else:
            self.L = L

        self.Nsum = None
        self.sbt = StickBreakingTransform(0)

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
            data['y'][i] = data['y'][i].reshape(self.N[i], self.J)
    
    def gen_default_priors(self, data, K, L,
                           # sig_prior=Gamma(10, 10),
                           sig_prior=LogNormal(0, 0.01),
                           alpha_prior=Gamma(1, 1),
                           mu0_prior=None,
                           mu1_prior=None,
                           W_prior=None,
                           eta0_prior=None,
                           eta1_prior=None):

        if L is None:
            L = [5, 5]

        self.__cache_model_constants__(data, K, L)

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
                       'eta0': eta0_prior, 'eta1': eta1_prior, 'W': W_prior,
                       'alpha': alpha_prior}

    def init_vp(self): 
        return {'mu0': VarParam(self.L[0]),
                'mu1': VarParam(self.L[1]),
                'sig': VarParam(self.I, init_m=0, init_log_s=-10),
                'W': VarParam((self.I, self.K - 1)),
                'v': VarParam(self.K),
                'alpha': VarParam(1),
                'H': VarParam((self.J, self.K)),
                'eta0': VarParam((self.I, self.J, self.L[0] - 1)),
                'eta1': VarParam((self.I, self.J, self.L[1] - 1))}

    def subsample_data(self, data, minibatch_info=None):
        if minibatch_info is None:
            mini_data = data
        else:
            mini_data = {'y': [], 'm': []}
            for i in range(self.I):
                n = int(minibatch_info['prop'] * self.N[i])
                idx = np.random.choice(self.N[i], n)
                mini_data['y'].append(data['y'][i][idx, :])
                mini_data['m'].append(data['m'][i][idx, :])
        return mini_data

    def sample_real_params(self, vp):
        real_params = {}
        for key in vp:
            real_params[key] = vp[key].sample()
        return real_params

    def log_q(self, real_params, vp):
        out = 0.0
        for key in vp:
            out += vp[key].log_prob(real_params[key]).sum()
        if self.debug:
            print('log_q: {}'.format(out / self.Nsum))
        return out / self.Nsum

    def log_prior(self, real_params):
        # FIXME. These should be ordered.
        lp_mu = 0.0
        for z in range(2):
            muz = 'mu0' if z == 0 else 'mu1'
            lp_mu += lpdf_logx(real_params[muz], self.priors[muz]).sum()

        # lp_sig = lpdf_logGamma(real_params['sig'], self.priors['sig']).sum()
        lp_sig = lpdf_logx(real_params['sig'], self.priors['sig']).sum()

        # ok when the last dimension is Dirichlet
        lp_W = lpdf_realDirichlet(real_params['W'], self.priors['W']).sum()

        lp_v = lpdf_logitBeta(real_params['v'],
                              Beta(real_params['alpha'].exp(), torch.tensor(1.0))).sum()

        lp_alpha = lpdf_logx(real_params['alpha'], self.priors['alpha']).sum()

        # H: J x K
        lp_H = Normal(0, 1).log_prob(real_params['H']).sum()

        # ok when the last dimension is Dirichlet
        lp_eta0 = lpdf_realDirichlet(real_params['eta0'], self.priors['eta0']).sum()
        lp_eta1 = lpdf_realDirichlet(real_params['eta1'], self.priors['eta1']).sum()
        lp_eta = lp_eta0 + lp_eta1

        # sum up the log priors
        lp = lp_mu + lp_sig + lp_W + lp_v + lp_alpha + lp_eta + lp_H

        if self.debug:
            print('log_prior:       {}'.format(lp / self.Nsum))
            # print('log_prior mu:    {}'.format(lp_mu))
            # print('log_prior sig:   {}'.format(lp_sig))
            # print('log_prior W:     {}'.format(lp_W))
            # print('log_prior v:     {}'.format(lp_v))
            # print('log_prior H:     {}'.format(lp_H))
            # print('log_prior alpha: {}'.format(lp_alpha))
            # print('log_prior eta:   {}'.format(lp_eta))

        return lp / self.Nsum

    def loglike(self, real_params, data, minibatch_info=None):
        params = self.to_param_space(real_params)
        # if self.debug:
        #     print(params)

        ll = 0.0
        
        # FIXME: Check this!
        for i in range(self.I):
            # Y: Ni x J
            # muz: Lz
            # etaz_i: 1 x J x Lz

            # Ni x J x Lz
            d0 = Normal(-self.iota - params['mu0'][None, None, :].cumsum(2),
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

            # OLD
            # log_b_vec = params['v'].log().cumsum(0)
            # Z = (log_b_vec[None, :] > Normal(0, 1).cdf(params['H']).log()).float()

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

    def to_real_space(self, params):
        return {'mu0': params['mu0'].log(),
                'mu1': params['mu1'].log(),
                'sig': params['sig'].log(),
                'W': self.sbt.inv(params['W']),
                'v': params['v'].log() - (-params['v']).log1p(),
                'H': params['H'],
                'alpha': params['alpha'].log(),
                'eta0': self.sbt.inv(params['eta0']),
                'eta1': self.sbt.inv(params['eta1'])}

    def to_param_space(self, real_params):
        return {'mu0': real_params['mu0'].exp(),
                'mu1': real_params['mu1'].exp(),
                'sig': real_params['sig'].exp(),
                'W': self.sbt(real_params['W']),
                'v': real_params['v'].sigmoid(),
                'H': real_params['H'],
                'alpha': real_params['alpha'].exp(),
                'eta0': self.sbt(real_params['eta0']),
                'eta1': self.sbt(real_params['eta1'])}

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

        optimizer = torch.optim.Adam([vp[key].m for key in param_names] + 
                                     [vp[key].log_s for key in param_names],
                                     lr=lr)
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
                    grad_m_isnan = torch.isnan(vp[key].m.grad)
                    if grad_m_isnan.sum() > 0:
                        print("WARNING: in {} m: {}".format(key, vp[key].m))
                        print("WARNING: in {} m.grad: {}".format(key, vp[key].m.grad))
                        print("WARNING: Setting a nan gradient to zero in {}!".format(key))
                        vp[key].m.grad[grad_m_isnan] = 0.0
                        fixed_grad = True

                    grad_log_s_isnan = torch.isnan(vp[key].log_s.grad)
                    if grad_log_s_isnan.sum() > 0:
                        print("WARNING: in {} log_s: {}".format(key, vp[key].log_s))
                        print("WARNING: in {} log_s.grad: {}".format(key, vp[key].log_s.grad))
                        print("WARNING: Setting a nan gradient to zero in {}!".format(key))
                        vp[key].log_s.grad[grad_log_s_isnan] = 0.0
                        fixed_grad = True

            if fixed_grad:
                for key in vp:
                    with torch.no_grad():
                        vp[key].m.data = best_vp[key].m.data
                        vp[key].log_s.data = best_vp[key].log_s.data

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

            if t > 100 and sum(math.isnan(eb) for eb in elbo[-10:]) == 10:
                print("ELBO is becoming nan. Terminating optimizer early.")
                self.vp = best_vp
                break

        return {'vp': vp, 'elbo': elbo, 'trace': trace}

