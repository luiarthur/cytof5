using Flux, Flux.Tracker
using Distributions

struct ModelParam
  m::TrackedArray
  log_s::TrackedArray
  support::String # real, unit, simplex, positive
  size::Integer
  
  function ModelParam(K::Integer, support::String)
    @assert(support in ["real", "unit", "simplex", "positive"])
    return new(param(randn(K)), param(randn(K)), support, K)
  end
end

"""
Get variational parameters
"""
function vp(mp::ModelParam)
  if mp.support in ["unit", "simplex"]
    m = sigmoid.(mp.m) .* 20.0 .- 10.0
    s = sigmoid.(mp.log_s) * 10.0
  elseif mp.support == "positive"
    m = exp.(mp.m)
    s = exp.(mp.log_s)
  else
    m = mp.m
    s = exp.(mp.log_s)
  end

  return (m, s)
end

"""
Reparameterized sampling from variational distribution
"""
function rsample(mp::ModelParam)
  m, s = vp(mp)
  return randn(mp.size) .* s .+ m
end

#= TEST
mp = ModelParam(3, "unit")
vp(mp)
rsample(mp)
=#

