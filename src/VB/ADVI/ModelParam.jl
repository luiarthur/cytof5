struct ModelParam{T, S <: NTuple{N, Int} where N}
  m::T
  log_s::T
  support::String # real, unit, simplex, positive
  size::S
end

# scalar param
function ModelParam(support::String; m=nothing, log_s=nothing)
  @assert(support in ["real", "unit", "simplex", "positive"])
  if m==nothing
    m = param(randn(1))
  end
  if log_s==nothing
    log_s = param(randn(1))
  end
  return ModelParam(m, log_s, support, (1, ))
end

# Vector param
function ModelParam(K::Integer, support::String; m=nothing, log_s=nothing)
  @assert(support in ["real", "unit", "simplex", "positive"])
  if m==nothing
    m = param(randn(K))
  end
  if log_s==nothing
    log_s = param(randn(K))
  end
  return ModelParam(m, log_s, support, (K, ))
end

# ND-Array param
function ModelParam(D::Tuple, support::String; m=nothing, log_s=nothing)
  @assert(support in ["real", "unit", "simplex", "positive"])
  if m==nothing
    m = param(randn(D...))
  end
  if log_s==nothing
    log_s = param(randn(D...))
  end
  return ModelParam(m, log_s, support, D)
end


function log_q(mp::ModelParam, r::R; args...) where R
  m, s = vp(mp; args...)
  return lpdf_normal.(r, m, s)
end

"""
Get variational parameters
"""
function vp(mp::ModelParam; m_min::Float64=-10.0, m_max::Float64=10.0, s_max::Float64=10.0)
  if mp.support in ("unit", "simplex")
    param_range = m_max - m_min
    m = sigmoid.(mp.m) .* param_range .+ m_min
    s = sigmoid.(mp.log_s) .* s_max
  else
    m = mp.m
    s = exp.(mp.log_s)
  end

  return (m, s)
end

function logabsdetJ(mp::ModelParam, real::R, tran::T) where {R, T}
  if mp.support == "simplex"
    return SB_logabsdetJ(real, tran)
  elseif mp.support == "unit"
    return log.(tran) + log1p.(-tran)
  elseif mp.support == "positive"
    return real
  elseif mp.support == "real"
    return zero(mp.m)
  else
    ErrorException("ADVI.ModelParam.logabsdet is not implemented for support=$(mp.support)")
  end
end

function transform(mp::ModelParam, real::T) where T
  if mp.support == "simplex"
    return SB_transform(real)
  elseif mp.support == "unit"
    return sigmoid.(real)
  elseif mp.support == "positive"
    return exp.(real)
  elseif mp.support == "real"
    return real
  else
    ErrorException("ADVI.ModelParam.transform is not implemented for support=$(mp.support)")
  end
end

"""
Reparameterized sampling from variational distribution
"""
function rsample(mp::ModelParam)
  # TODO: Optimize this
  m, s = vp(mp)
  return randn(mp.size) .* s .+ m
end

#= TEST
mp = ModelParam(3, "unit")
vp(mp)
rsample(mp)
=#

"""
Get variational parameters
"""
vparams(mp::ModelParam) = Flux.params(mp.m, mp.log_s)

function vparams(s::S) where S
  ps = []
  for key in fieldnames(S)
    if isdefined(s, key)
      f = getfield(s, key)
      if typeof(f) <: ModelParam
        append!(ps, [f.m])
        append!(ps, [f.log_s])
      elseif Tracker.istracked(f)
        append!(ps, [f])
      end
    else
      @warn "Field $key is undefined in $(S.name)"
    end
  end
  return Flux.params(ps...)
end
