module Bla
using Flux, Flux.Tracker
using Distributions

# FIXME
struct ModelParam{T, S <: Union{Integer, Vector{Integer}}}
  m::T
  log_s::T
  support::String # real, unit, simplex, positive
  size::S
end

# TR(T) = typeof(param(rand(T)))
# TV(T) = typeof(param(rand(T, 0)))
# TM(T) = typeof(param(rand(T, 0)))

function ModelParam(T::Type, K::Integer, support::String)
  @assert(support in ["real", "unit", "simplex", "positive"])
  return ModelParam(param(randn(T, K)), param(randn(T, K)), support, K)
end

function ModelParam(T::Type, D::Array{Int64, 1}, support::String)
  @assert(support in ["real", "unit", "simplex", "positive"])
  return ModelParam(param(randn(T, D...)), param(randn(T, D...)), support, D)
end
end # BLA

#= Test
v = Bla.ModelParam(Float32, 3, "unit")
a = Bla.ModelParam(Float32, [3, 5], "unit")
=#

"""
Get variational parameters
"""
function vp(mp::ModelParam)
  if mp.support in ["unit", "simplex"]
    m = sigmoid.(mp.m) .* 20.0 .- 10.0
    s = sigmoid.(mp.log_s) * 10.0
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

