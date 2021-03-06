println("pid: $(getpid())")

using PyCall
using BSON

# pyimports
plt = pyimport("matplotlib.pyplot")
cytofpy = pyimport(:cytofpy)
torch = pyimport(:torch)
Gamma = torch.distributions.Gamma
Beta = torch.distributions.Beta

SIMDAT_PATH = "../../../sims/sim_study/simdata/kills-flowsom/N500/K5/90/simdat.bson"

simdat = BSON.load(SIMDAT_PATH)[:simdat]
y = [torch.tensor(yi) for yi in simdat[:y]]
priors = cytofpy.model.default_priors(y, K=30, L=[5, 3],
                                      y_quantiles=[0.02, .25, .48],
                                      p_bounds=[.05, .8, .05])
priors["sig2"] = Gamma(.1, 1)
priors["alpha"] = Gamma(.1, .1)
priors["delta0"] = Gamma(1, 1)
priors["delta1"] = Gamma(1, 1)
priors["noisy_var"] = 10.0
priors["eps"] = Beta(100, 9900)

every(max_iter, nsave) = round(Int, max_iter / nsave)
max_iter = 10000
nsave = 30
out = cytofpy.model.fit(y, max_iter=max_iter, lr=1e-2, print_freq=10, eps_conv=0,
                        priors=priors, minibatch_size=500, tau=0.005,
                        trace_every=50, backup_every=every(max_iter, nsave),
                        verbose=0, seed=1, use_stick_break=false)

out = Dict(Symbol(k) => v for (k, v) in out)
# BSON.bson("results/out.bson", out)

#= Plots
import Distributions
out = BSON.load("results/out.bson")
mp = out[:mp]
v = mp["v"]
H = mp["H"]
Z = [v.dist().rsample().reshape(1, priors["K"]) > H.dist().rsample() for i in 1:100]
Z_mean = Distributions.mean(Z)
eps_samps = [mp["eps"].transform(mp["eps"].dist().rsample()) for i in 1:100]

plt.plot(out[:elbo]); plt.show()
plt.imshow(Z_mean.numpy()); plt.show()
=#
