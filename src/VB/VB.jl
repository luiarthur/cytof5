module VB

using Distributions
using Flux, Flux.Tracker

import Random # suffle, seed

include("State.jl")
include("ModelParam.jl")

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


function loglike(params, c, data, idx)
  y = params[:y]
  sig = sqrt.(params[:sig2])

  ll = zero(params[:alpha])
  for i in 1:c.I
    # Ni x J x Lz
    yi = reshape(y[i], c.N[i], c.J, 1)

    mu0 = reshape(-cumsum(params[:delta0], 1, 1, c.L[0]))
    lf0 = logpdf.(Normal(mu0, sig[i]), yi) .+ log.(params[:eta0][i:i, :, :])

    mu1 = reshape(cumsum(params[:delta1], 1, 1, c.L[1]))
    lf1 = logpdf.(Normal(mu1, sig[i]), yi) .+ log.(params[:eta1][i:i, :, :])

    # Ni x J
    logmix_L0 = logsumexp(lf0, dims=3)
    logmix_L1 = logsumexp(lf1, dims=3)

    # Z: J x K
    # H: J x K
    # v: K
    # c: Ni x J x K
    # d: Ni x K
    # Ni x J x K

    if c.use_stickbreak
      v = cumprod(params[:v])
    else
      v = params[:v]
    end


  end

  println("NotImplemented")
  return ll
end


function logprior(reals, params)
  println("NotImplemented")
end


function logq(reals)
  println("NotImplemented")
end

end # VB
