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

function sim_data(; N=100, mu=[-2, 3], sig=[.2, .4], w=[.2, .8], seed=missing)
  if !ismissing(seed)
    Random.seed!(seed)
  end

  K = length(mu)
  @assert K == length(sig) == length(w)
  @assert sum(w) == 1

  cluster_labels = wsample(1:K, w, N)
  y = randn(N) .* sig[cluster_labels] + mu[cluster_labels]
 
  return Dict(:cluster_labels => cluster_labels,
              :mu => mu, :sig => sig, :w => w, :y => y)
end

### MAIN ###

# Simulate data
simdat = sim_data(N=300)
N = length(simdat[:y])
K = 2

# Log posterior
function logpost(s::State)
  K = length(s.mu)
  mu = reshape(s.mu, 1, K)
  sig = reshape(exp.(s.log_sig), 1, K)
  # w = SB_transform(s.stickbreak_w)  # FIXME: I expected this would work.
  real_w = reshape(s.stickbreak_w, 1, K - 1)
  w = SB_transform(real_w)

  ll = (ADVI.lpdf_normal.(reshape(simdat[:y], N, 1), mu, sig) .+
        log.(w))
  ll = sum(ADVI.logsumexp(ll, dims=2))

  lp = (sum(ADVI.compute_lpdf(Normal(0, 3), mu)) +
        sum(ADVI.compute_lpdf(LogNormal(0, 1), sig)) +
        sum(ADVI.compute_lpdf(Dirichlet(K, K), w)))

  lpabsjacobian = (sum(s.log_sig) +
                   sum(SB_logabsdetJ(real_w, w)))

  return ll + lp + lpabsjacobian
end

state = State(K)
_ = logpost(state)

function simulate(init; nburn, nsamps, eps, num_leapfrog_steps, 
                  kappa=nothing, momentum_sd=1.0)
  state = deepcopy(init)

  samps = [state for i in 1:nsamps]
  log_prob_hist = zeros(nsamps)

  for i in 1:(nburn + nsamps)
    print("\rProgress: $i / $(nburn + nsamps)")
    eta = (kappa == nothing ? eps : eps * i ^ -kappa)
    state, curr_log_prob = HMC.hmc_update(state, logpost,
                                          num_leapfrog_steps, eta,
                                          momentum_sd=momentum_sd)
    if i > nburn
      samps[i - nburn] = state
      log_prob_hist[i - nburn] = curr_log_prob
    end
  end
  println()

  return Dict(:samps => samps, :log_prob => log_prob_hist)
end


# Compile
_ = simulate(state, nburn=1, nsamps=1, num_leapfrog_steps=1, eps=.1, kappa=.7)

# Simulate
@time out = simulate(state, nburn=500, nsamps=300,
                     num_leapfrog_steps=2^5, eps=1/N)

# FIXME: runs. but can't get correct answer...
println(simdat[:mu], simdat[:sig], simdat[:w])
println()

println("mu post mean: $(mean([o.mu for o in out[:samps]]))")
println("sig post mean: $(mean([exp.(o.log_sig) for o in out[:samps]]))")
println("w post mean: $(mean([SB_transform(o.stickbreak_w) for o in out[:samps]]))")

R"plot($(out[:log_prob]), type='l')"
end # module GmmHmcTest
