import abc # abstract base class

class Model(abc.ABC):
    def __init__(self, data, minibatch_size=0, lr=1e-3):
        pass

    @abc.abstractmethod
    def loglike(self, params):
        pass

    @abc.abstractmethod
    def logprior(self, params):
        """
        Log prior of parameters transformed to have real support.
        
        That is if p ~ Unif(a, b), which has support on the unit interval, then
        `logprior(self, p, var_params)` is the log prior for logit(p),
        which is on the real line.

        Arguments: 
        - params: dictionary containing all model the parameters
        - var_params: dictionary containing all variational parameters
        """
        pass

    @abc.abstractmethod
    def logq(self, params, var_params):
        """
        variational distribution in R^(dim(params))
        """
        pass
