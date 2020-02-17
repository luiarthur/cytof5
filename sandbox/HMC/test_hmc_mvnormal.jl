import Pkg; Pkg.activate("../../")  # Cytof5

# See 4.1.1 of http://www.stat.columbia.edu/~gelman/research/published/nuts.pdf
include("HMC.jl")
import LinearAlgebra
using Distributions, Random
using Flux
using Flux.Tracker

using RCall
@rimport graphics as rgraphics
@rimport rcommon

eye(T, n::Int) = Matrix{T}(LinearAlgebra.I, n, n)
eye(n::Int) = Matrix{Float64}(LinearAlgebra.I, n, n)
prettymat(X) = begin show(stdout, "text/plain", X); println(); end

struct MX
  x
end

MX(K::Int) = MX(param(randn(K)))

function gen_distribution(P)
  S = rand(InverseWishart(P, eye(Float64, P)))
  m = zeros(P)
  return (m, S)
end

### MAIN ###
Random.seed!(0)

easy = true  # bivariate normal
# easy = false  # 250-variate normal

if easy 
  K = 3
  S = eye(K)
  S[1,2] = S[2,1] = -0.8
  S[1,3] = S[3,1] = 0.5
else
  K = 250
  S = rand(InverseWishart(K, eye(K)))
end
inv_S = inv(S)
log_prob(s::MX) = -s.x' * inv_S * s.x / 2.0
mvn = MvNormal(S)


function simulate(init; nburn, nsamps, eps, num_leapfrog_steps, 
                  kappa=nothing, momentum_sd=1.0)
  state = deepcopy(init)

  samps = [state for i in 1:nsamps]
  log_prob_hist = zeros(nsamps)

  for i in 1:(nburn + nsamps)
    print("\rProgress: $i / $(nburn + nsamps)")
    eta = (kappa == nothing ? eps : eps * i ^ -kappa)
    state, curr_log_prob = HMC.hmc_update(state, log_prob,
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

# initial state
state = MX(param(zeros(K)))

# Compile
_ = simulate(state, nburn=1, nsamps=1, num_leapfrog_steps=1, eps=.1, kappa=.7)

# Simulate
@time if easy
  out = simulate(state, nburn=1000, nsamps=10000,
                 num_leapfrog_steps=2^5, eps=.05)
  # out = simulate(state, nburn=10000, nsamps=10000,
  #                num_leapfrog_steps=2^2, eps=.05)
else
  out = simulate(state, nburn=1000, nsamps=1000,
                 num_leapfrog_steps=2^8, eps=.001)
end
samps = out[:samps]
log_prob_hist = out[:log_prob]
acceptance_rate = (length(unique(log_prob_hist)) - 1) / length(log_prob_hist)
println("acceptance rate: $(acceptance_rate)")

# Plot
post_x = Matrix(hcat([Tracker.data(s.x) for s in samps]...)')
prettymat(cor(mvn))
prettymat(cor(post_x))

rgraphics.plot(log_prob_hist, xlab="iteration", ylab="log pdf",
               main="trace", typ="l");
rcommon.plotPosts(post_x[:, 1:3]);

