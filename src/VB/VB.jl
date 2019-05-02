module VB

using Distributions
using Flux, Flux.Tracker

import Random # shuffle, seed

include("ADVI/ADVI.jl")
include("vae.jl")
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
function compute_Z(v::AbstractArray, H::AbstractArray;
                   use_stickbreak::Bool=false, tau::T=.001) where T
  v_rs = reshape(v, 1, length(v))
  p = use_stickbreak ? cumprod(v_rs, dims=1) : v_rs
  logit = p .- H
  smoothed_Z = sigmoid.(logit / tau)
  Z = (smoothed_Z .> T(0.5)) * T(1.0)
  return (Z - smoothed_Z).data + smoothed_Z
end


function prob_miss_logit(y::S, b0::T, b1::T, b2::T) where {S, T}
  return b0 .+ b1 .* y .+ b2 .* y .^ 2
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
