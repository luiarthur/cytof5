module VB

using Distributions
using Flux, Flux.Tracker

import Random # suffle, seed

include("ModelParam.jl")
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
function compute_Z(logit::T, tau::Float64=.001) where T
  smoothed_Z = sigmoid.((logit / tau))
  Z = (smoothed_Z .> 0.5) * 1.0
  return (Z - smoothed_Z).data + smoothed_Z
end


function prob_miss_logit(y::S, b0::T, b1::T, b2::T) where {S, T}
  return b0 .+ b1 .* y .+ b2 .* y .^ 2
end

include("loglike.jl")

function logprior(reals, params)
  println("NotImplemented")
end


function logq(reals)
  println("NotImplemented")
end

end # VB
