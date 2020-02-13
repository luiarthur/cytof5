module GmmHmcTest
import Pkg; Pkg.activate("../../")  # Cytof5

include("HMC.jl")
using Distributions, Random
using Flux, Flux.Tracker
using Flux.Tracker: @grad, track, lgamma
include("../../src/VB/ADVI/ADVI.jl")
include("../../src/VB/ADVI/custom_grads.jl")
include("../../src/VB/ADVI/custom_functions.jl")
include("../../src/VB/ADVI/StickBreak.jl")

using RCall
@rimport graphics as rgraphics
@rimport rcommon

Random.seed!(0)

struct State
  mu
  log_sig
  stickbreak_w
end

pretty(s::State) = ("State(μ => $(s.mu), " * 
                    "σ => $(exp.(s.log_sig)), " *
                    "w => $(SB_transform(s.stickbreak_w)))")

State(K::Int) = State(param(randn(K)),  # mu
                      param(randn(K)),  # sig
                      param(randn(K - 1)))  # stickbreak_w

function sim_data(; N=100, mu=[-1, 1], sig=[1, .5], w=[.3, .7], seed=missing)
  if !ismissing(seed)
    Random.seed(seed)
  end

  K = length(mu)
  @assert K == length(sig) == length(w)
  @assert sum(w) == 1

  cluster_labels = wsample(1:K, w, N)
  mu = mu[cluster_labels]
  sig = sig[cluster_labels]
  y = randn(N) .* sig + mu
 
  return Dict(:cluster_labels => cluster_labels,
              :mu => mu, :sig => sig, :w => w, :y => y)
end

### MAIN ###

# Simulate data
simdat = sim_data()
N = length(simdat[:y])
K = 3

# Log posterior
function logpost(s::State)
  K = length(s.mu)
  mu = reshape(s.mu, 1, K)
  sig = reshape(exp.(s.log_sig), 1, K)
  w = SB_transform(s.stickbreak_w)

  ll = (ADVI.lpdf_normal.(reshape(simdat[:y], N, 1), mu, sig) .+
        log.(reshape(w, 1, K)))
  ll = sum(ADVI.logsumexp(ll, dims=2))

  lp = (sum(ADVI.compute_lpdf(Normal(0, 1), mu)) +
        sum(ADVI.compute_lpdf(LogNormal(0, 1), sig)) +
        sum(ADVI.compute_lpdf(Dirichlet(K, 1 / K), w)))

  a = sum(s.log_sig)
  b = sum(SB_logabsdetJ(s.stickbreak_w, w))

  lpabsjacobian = (sum(s.log_sig) +
                   sum(SB_logabsdetJ(s.stickbreak_w, w)))

  return ll + lp + lpabsjacobian
end

state = State(K)
_ = logpost(state)
num_leapfrog_steps = 10
eps = .01

# FIXME?
_ = HMC.hmc_update(state, logpost, num_leapfrog_steps, eps)



end # module GmmHmcTest

# Maybe try this in python???
