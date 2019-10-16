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

struct State
  x
end

State(K::Int) = State(param(randn(K)))

function gen_distribution(P)
  S = rand(Wishart(P, eye(Float64, P)))
  m = zeros(P)
  return MvNormal(m, S)
end

function logpdf_mvn(x, m, S)
  y = x - m
  return -y'S*y / 2.0
end

### MAIN ###
Random.seed!(0)

K = 250
mvn = gen_distribution(K)
log_prob(s::State) = logpdf_mvn(s.x, mvn.μ, Matrix(mvn.Σ))


function simulate(init; nburn, nsamps, eps, num_leapfrog_steps)
  state = deepcopy(init)

  samps = [state for i in 1:nsamps]
  for i in 1:(nburn + nsamps)
    print("\rProgress: $i / $(nburn + nsamps)")
    state = HMC.hmc_update(state, log_prob, num_leapfrog_steps, eps)
    if i > nburn
      samps[i - nburn] = state
    end
  end

  return samps
end

# initial state
state = State(K)

# Compile
_ = simulate(state, nburn=1, nsamps=1, num_leapfrog_steps=1, eps=.1)

# Simulate
samps = simulate(state, nburn=500, nsamps=500, num_leapfrog_steps=30, eps=.1)

# Plot
head_x = Matrix(hcat([Tracker.data(s.x[1:2]) for s in samps]...)')
rcommon.plotPosts(head_x)
