function head(x::T) where {T <: AbstractArray}
  """
  does the same thing as `x[..., :-1]` in pytorch.
  """
  return x[[axes(x, i) for i in 1:ndims(x)-1]..., 1:end-1]
end

function slice(x::T, l::Integer) where {T <: AbstractArray}
  """
  returns: x[[axes(x, i) for i in 1:ndims(x)-1]..., l]
  """
  return x[[axes(x, i) for i in 1:ndims(x)-1]..., l]
end

function tail(x::T) where {T <: AbstractArray}
  """
  returns: x[[axes(x, i) for i in 1:ndims(x)-1]..., end]
  """
  return x[[axes(x, i) for i in 1:ndims(x)-1]..., end]
end

function logsumexp(logx::T; dims::Integer=1) where {T <: AbstractArray}
  """
  logsumexp
  """
  mx = maximum(logx, dims=dims)
  return mx .+ log.(sum(exp.(logx .- mx), dims=dims))
end

function lpdf_normal(x::X, m::M, s::S) where {X <: Real, M <: Real, S<:Real}
  z = (x - m) / s
  return oftype(z, -0.5 * log(2*pi) - z^2 * 0.5 - log(s))
end

function lpdf_gmm(x::TX, m::TM, s::TS, w::TW; dims::Integer) where {TX, TW, TM, TS}
  """
  log pdf of gaussian mixture model
  """
  return logsumexp(log.(w) .+ lpdf_normal.(x, m, s), dims=dims)
end
