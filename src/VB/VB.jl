module VB

using Distributions
import Random # suffle, seed
using Flux, Flux.Tracker

"""
This enables the backward gradient computations via Z.
Notice at the end, we basically return Z (binary tensor).
But we make use of the smoothed Z, which is differentiable.
We detach (Z - smoothed_Z) so that the gradients are not 
computed, and then add back smoothed_Z for the return.
The only gradient will then be that of smoothed Z.
"""
function compute_Z(logit::T, tau::Float64) where T
  smoothed_Z = sigmoid.((logit / tau))
  Z = (smoothed_Z .> 0.5) * 1.0
  return (Z - smoothed_Z).data + smoothed_Z
end


function prob_miss_logit(y::S, b0::T, b1::T, b2::T) where {S, T}
  return b0 .+ b1 .* y .+ b2 .* y .^ 2
end


# TODO
function loglike(s, c, d)
  y = s.y
  sig = sqrt.(s.sig2)
  ll = Tracker.TrackedReal(0.)

  for i in 1:c.I
    mu0 = -cumsum(s.delta0, 1)
    mu1 = cumsum(s.delta1, 1)
    # TODO
  end

  return ll
end

end # VB
