module VB

using Distributions
using Flux, Flux.Tracker

import Random # shuffle, seed

include("ADVI/ADVI.jl")
include("State.jl")
include("Constants.jl")

"""
This enables the backward gradient computations via Z.
Notice at the end, we basically return Z (binary tensor).
But we make use of the smoothed Z, which is differentiable.
We detach (Z - smoothed_Z) so that the gradients are not 
computed, and then add back smoothed_Z for the return.
The only gradient will then be that of smoothed Z.
"""
# function compute_Z(logit::T, tau::Float64=.001) where T
function compute_Z(v::AbstractArray, H::AbstractArray, tau::Float64=.001) where T
  logit = v .- H
  smoothed_Z = sigmoid.((logit / tau))
  Z = (smoothed_Z .> 0.5) * 1.0
  return (Z - smoothed_Z).data + smoothed_Z
end


function prob_miss_logit(y::S, b0::T, b1::T, b2::T) where {S, T}
  return b0 .+ b1 .* y .+ b2 .* y .^ 2
end

function lpdf_normal(x::X, m::M, s::S) where {X <: Real, M <: Real, S<:Real}
  z = (x - m) / s
  return -0.5 * log(2*pi) - z^2 * 0.5 - log(s)
end

include("loglike.jl")
include("logprior.jl")
include("logq.jl")

function compute_elbo(state, y::Vector{Matrix{AbstractFloat}}, c::Constants)
  real, tran = rsample(state);
  # TODO
  ll = loglike(tran, y, c)
  lp = logprior(real, tran, c)
  lq = logq(real, c)
  elbo = ll + lp - lq
  return elbo / sum(c.N)
end

end # VB
