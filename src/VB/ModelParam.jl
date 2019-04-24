using Flux, Flux.Tracker
using Distributions

include("StickBreak/StickBreak.jl")
const SB = StickBreak

struct ModelParam{T, ET, S <: NTuple{N, Int} where N}
  m::T
  log_s::T
  support::String # real, unit, simplex, positive
  size::S
  eltype::Type{ET}
end

MPA{F, N} = ModelParam{TrackedArray{F, N, Array{F, N}}, F, NTuple{N, Int}}
MPR{F} = ModelParam{Tracker.TrackedReal{F}, F, NTuple{0, Int}}

# scalar param
function ModelParam(T::Type, support::String)
  @assert(support in ["real", "unit", "simplex", "positive"])
  return ModelParam(param(randn(T)), param(randn(T)), support, (), T)
end

# Vector param
function ModelParam(ElType::Type, K::Integer, support::String)
  @assert(support in ["real", "unit", "simplex", "positive"])
  return ModelParam(param(randn(ElType, K)), param(randn(ElType, K)), support, (K, ), ElType)
end

# ND-Array param
function ModelParam(ElType::Type, D::Tuple, support::String)
  @assert(support in ["real", "unit", "simplex", "positive"])
  return ModelParam(param(randn(ElType, D...)), param(randn(ElType, D...)), support, D, ElType)
end


"""
Get variational parameters
"""
function vp(mp::ModelParam; m_min::Float64=-10.0, m_max::Float64=10.0, s_max::Float64=10.0)
  if mp.support in ["unit", "simplex"]
    param_range = mp.eltype(m_max - m_min)
    m = sigmoid.(mp.m) .* param_range .+ mp.eltype(m_min)
    s = sigmoid.(mp.log_s) .* mp.eltype(s_max)
  else
    m = mp.m
    s = exp.(mp.log_s)
  end

  return (m, s)
end

function logabsdetJ(mp::ModelParam, real::R, tran::T) where {R, T}
  if mp.support == "simplex"
    println("NotImplemented")
  elseif mp.support == "unit"
    println("NotImplemented")
  elseif mp.support == "positive"
    println("NotImplemented")
  else # "real"
    return zero(mp.m)
  end
end

function transform(mp::ModelParam, real::T) where T
  if mp.support == "simplex"
    return SB.transform(real)
  elseif mp.support == "unit"
    return one(mp.eltype) ./ (one(mp.eltype) .+ exp.(real))
  elseif mp.support == "positive"
    return exp.(real)
  else # "real"
    return real
  end
end

"""
Reparameterized sampling from variational distribution
"""
function rsample(mp::ModelParam)
  # TODO: Optimize this
  m, s = vp(mp)
  if mp.size == 0
    return randn(mp.eltype) * s + m
  else
    return randn(mp.eltype, mp.size) .* s .+ m
  end
end

#= TEST
mp = ModelParam(3, "unit")
vp(mp)
rsample(mp)
=#

