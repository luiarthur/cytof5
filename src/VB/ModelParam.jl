using Flux, Flux.Tracker
using Distributions

include("StickBreak.jl")
const SB = StickBreak

struct ModelParam{T, S <: Union{Tuple, Integer}}
  m::T
  log_s::T
  support::String # real, unit, simplex, positive
  size::S
end

# TS(T) = typeof(param(rand(T)))
# TV(T) = typeof(param(rand(T, 0)))
# TM(T) = typeof(param(rand(T, 0)))

# scalar param
function ModelParam(T::Type, support::String)
  @assert(support in ["real", "unit", "simplex", "positive"])
  return ModelParam(param(randn(T)), param(randn(T)), support, 0)
end

# Vector param
function ModelParam(ElType::Type, K::Integer, support::String)
  @assert(support in ["real", "unit", "simplex", "positive"])
  return ModelParam(param(randn(ElType, K)), param(randn(ElType, K)), support, K)
end

# ND-Array param
function ModelParam(ElType::Type, D::S, support::String) where {S <: Tuple}
  @assert(support in ["real", "unit", "simplex", "positive"])
  return ModelParam(param(randn(ElType, D...)), param(randn(ElType, D...)), support, D)
end


"""
Get variational parameters
"""
function vp(mp::ModelParam)
  # TODO:
  # make the values non-hardcoded
  if mp.support in ["unit", "simplex"]
    m = sigmoid.(mp.m) .* 20.0 .- 10.0
    s = sigmoid.(mp.log_s) .* 10.0
  else
    m = mp.m
    s = exp.(mp.log_s)
  end

  return (m, s)
end

function transform(mp::ModelParam, real::T) where T
  if mp.support == "simplex"
    return SB.transform(real)
  elseif mp.support == "unit"
    return 1.0 ./ (1.0 .+ exp.(real))
  elseif mp.support == "positive"
    return exp.(real)
  else mp.support # "real"
    return real
  end
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

