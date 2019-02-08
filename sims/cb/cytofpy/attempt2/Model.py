import abc
import copy
import datetime
import math
import torch

class Model(abc.ABC):
    def __init__(self, data, priors, dtype=torch.float64, device="cpu", misc=None):
        self.dtype = dtype
        self.device = device
        self.data = data
        self.priors = priors
        self.misc = misc
        self.vp = None


    @abc.abstractmethod
    def loglike(self, data, params, minibatch_info):
        """
        log likelihood of parameters in original space

        params: parameters in original parameter space
        minibatch_info (dict): information related to minibatch. Used to
                               compute loglike if a minibatch is used.
        """
        pass

    def sample_params(self, vp=None):
        """
        samples parameters in real space from the variational distributions
        given the variational parameters in real space.
        """
        params = {}

        if vp is None:
            for key in self.vp:
                params[key] = self.vp[key].sample()
        else:
            for key in vp:
                params[key] = vp[key].sample()

        return params

    @abc.abstractmethod
    def subsample_data(self, minibatch_info=None):
        """
        subsample data
        """
        pass

    @abc.abstractmethod
    def init_vp(self):
        """
        initialize variational parameters in real space
        """
        pass

    @abc.abstractmethod
    def kl_qp(self, params):
        pass

    def compute_elbo(self, data, minibatch_info=None):
        """
        compute elbo

        vp: variational parameters
        data: data (may be a minibatch)
        minibatch_info: Information about minibatch
        """
        params = self.sample_params()
        ll = self.loglike(data=data, params=params, minibatch_info=minibatch_info)
        res = ll - self.kl_qp(params)
        assert(res.dim() == 0)
        return res

    def compute_elbo_mean(self, nmc, minibatch_info):
        """
        Compute the mean of the elbo via Monte Carlo integration
        
        The number of MC samples (nmc) can be as little as 1 in practice.
        But for some models, nmc=2 may be necessary. nmc >= 10 could be
        overkill for most if not all problems.

        data: data
        vp: variational parameters (real space)
        nmc: number of MC samples
        minibatch_info: information about minibatch
        """
        mini_data = self.subsample_data(minibatch_info)
        return torch.stack([self.compute_elbo(mini_data, minibatch_info)
                            for i in range(nmc)]).mean(0)

    def msg(self, t):
        """
        an optional message to print at the end of each iteration.

        t: iteration number
        vp: variational parameters (real space)
        """
        pass

    def vp_as_list(self):
        return [self.vp[key].vp for key in self.vp]
        
    def fit(self, niters:int=1000, nmc:int=2, lr:float=1e-2,
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
        torch.manual_seed(seed)


        if init is not None:
            self.vp = copy.deepcopy(init)
        else:
            self.init_vp()
        
        best_vp = copy.deepcopy(self.vp)
        vp_list = self.vp_as_list()

        optimizer = torch.optim.Adam(vp_list, lr=lr)
        elbo = []
        trace = []

        for t in range(niters):
            elbo_mean = self.compute_elbo_mean(nmc, minibatch_info)
            loss = -elbo_mean
            optimizer.zero_grad()
            loss.backward(retain_graph=True)

            repaired_grads = False
            with torch.no_grad():
                for key in self.vp:
                    grad_isnan = torch.isnan(self.vp[key].vp.grad)
                    if grad_isnan.sum() > 0:
                        print("WARNING: Setting a nan gradient to zero in {}.".format(key))
                        self.vp[key].vp.grad[grad_isnan] = 0.0
                        self.vp[key].vp.data = best_vp[key].vp.data
                        repaired_grads = True

            if t % 10 == 0 and not repaired_grads:
                best_vp = copy.deepcopy(self.vp)
                trace.append(best_vp)

            optimizer.step()
            elbo.append(elbo_mean.item())

            if print_freq > 0 and (t + 1) % print_freq == 0:
                now = datetime.datetime.now().replace(microsecond=0)
                if verbose >= 1:
                    print('{} | iteration: {}/{} | elbo: {}'.format(now, t + 1, niters, elbo[-1]))
                    
                self.msg(t)

            if t > 0 and abs(elbo[-1] / elbo[-2] - 1) < eps:
                print("Convergence suspected. Ending optimizer early.")
                break

            if t > 20 and sum(math.isnan(eb) for eb in elbo[-5:]) == 5:
                print("ELBO is becoming nan. Terminating optimizer early.")
                self.vp = best_vp
                break

        return {'vp': self.vp, 'elbo': elbo, 'trace': trace}

