using PyCall

# pyimports
cytofpy = pyimport(:cytofpy)
torch = pyimport(:torch)
Gamma = torch.distributions.Gamma
Beta = torch.distributions.Beta

y = [torch.randn(100,20) for i in 1:3]
priors = cytofpy.model.default_priors(y, K=30, L=[3, 5],
                                      y_quantiles=[0.0, .25, .5], p_bounds=[.05, .8, .05])
priors["sig2"] = Gamma(.1, 1)
priors["alpha"] = Gamma(.1, .1)
priors["delta0"] = Gamma(1, 1)
priors["delta1"] = Gamma(1, 1)
priors["noisy_var"] = 10.0
priors["eps"] = Beta(1, 99)

out = cytofpy.model.fit(y, max_iter=10000, lr=1e-2, print_freq=10, eps_conv=0,
                        priors=priors, minibatch_size=2000, tau=0.001,
                        trace_every=50, backup_every=50,
                        verbose=0, seed=1, use_stick_break=false)

