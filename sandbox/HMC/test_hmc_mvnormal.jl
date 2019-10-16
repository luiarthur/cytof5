#=
using Revise
=#

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
  K = 2
  S = eye(2); S[1,2] = S[2,1] = -0.8
else
  K = 250
  S = rand(InverseWishart(K, eye(K)))
end
inv_S = inv(S)
log_prob(s::MX) = -s.x' * inv_S * s.x / 2.0
mvn = MvNormal(S)


function simulate(init; nburn, nsamps, eps, num_leapfrog_steps, kappa=nothing)
  state = deepcopy(init)

  samps = [state for i in 1:nsamps]
  log_prob_hist = zeros(nsamps)

  for i in 1:(nburn + nsamps)
    print("\rProgress: $i / $(nburn + nsamps)")
    eta = (kappa == nothing ? eps : eps * i ^ -kappa)
    state, curr_log_prob = HMC.hmc_update(state, log_prob,
                                          num_leapfrog_steps, eta)
    if i > nburn
      samps[i - nburn] = state
      log_prob_hist[i - nburn] = curr_log_prob
    end
  end

  return Dict(:samps => samps, :log_prob => log_prob_hist)
end

# initial state
state = MX(param(zeros(K)))

# Compile
_ = simulate(state, nburn=1, nsamps=1, num_leapfrog_steps=1, eps=.1, kappa=.7)

# Simulate
out = simulate(state, nburn=100, nsamps=500, num_leapfrog_steps=100, eps=.01)
samps = out[:samps]
log_prob_hist = out[:log_prob]
acceptance_rate = (length(unique(log_prob_hist)) - 1) / length(log_prob_hist)
println("acceptance rate: $(acceptance_rate)")

# Plot
rgraphics.plot(log_prob_hist, xlab="iteration", ylab="log pdf",
               main="trace", typ="l");
post_x = Matrix(hcat([Tracker.data(s.x) for s in samps]...)')
head_x = post_x[:, 1:2]
rcommon.plotPosts(head_x);
cor(mvn)
cor(post_x)
