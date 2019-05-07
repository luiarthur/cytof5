module VB

using Distributions
using Flux, Flux.Tracker

import Random # shuffle, seed

include("ADVI/ADVI.jl")
include("vae.jl")
include("Priors.jl")
include("Constants.jl")
include("State.jl")

"""
This enables the backward gradient computations via Z.
Notice at the end, we basically return Z (binary tensor).
But we make use of the smoothed Z, which is differentiable.
We detach (Z - smoothed_Z) so that the gradients are not 
computed, and then add back smoothed_Z for the return.
The only gradient will then be that of smoothed Z.
"""
function compute_Z(v::AbstractArray, H::AbstractArray;
                   use_stickbreak::Bool=false, tau::Float64=.001)
  v_rs = reshape(v, 1, length(v))
  p = use_stickbreak ? cumprod(v_rs, dims=2) : v_rs
  logit = p .- H
  smoothed_Z = sigmoid.(logit / tau)
  Z = (smoothed_Z.data .> 0.5) # becomes non-tracked
  return (Z - smoothed_Z.data) + smoothed_Z
end

# function prob_miss(y::R, beta::AbstractFloat...) where {R <: Real}
#   n = length(beta)
#   x = sum([y^(i-1) * beta[i] for i in 1:n])
#   return sigmoid(x)
# end

function prob_miss(y::F, b0::Float64, b1::Float64, b2::Float64) where {F <: AbstractFloat}
  return sigmoid(b0 + b1 * y + b2 * y ^ 2)
end

function prob_miss(y::A, b0::Float64, b1::Float64, b2::Float64) where {A <: AbstractArray}
  return sigmoid.(b0 .+ b1 .* y .+ b2 .* y .^ 2)
end

include("loglike.jl")
include("logprior.jl")
include("logq.jl")

function compute_elbo(state::StateMP, y::Vector{M}, c::Constants,
                      metrics::Dict{Symbol, Vector{Float64}}) where M
  real, tran, yout, log_qy = rsample(state, y, c);

  m = [isnan.(yi) for yi in y]
  ll = loglike(tran, yout, m, c)
  lp = logprior(real, tran, state, c)
  lq = logq(real, state) + log_qy
  elbo = ll + lp - lq

  # store metrics
  append!(metrics[:ll], ll.data)
  append!(metrics[:lp], lp.data)
  append!(metrics[:lq], lq.data)
  append!(metrics[:elbo], elbo.data)

  return elbo
end

end # VB
