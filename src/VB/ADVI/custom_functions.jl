"""
does the same thing as `x[..., :-1]` in pytorch.
"""
function head(x::T) where {T <: AbstractArray}
  return x[[axes(x, i) for i in 1:ndims(x)-1]..., 1:end-1]
end

"""
returns: x[[axes(x, i) for i in 1:ndims(x)-1]..., l]
"""
function slice(x::T, l::Integer) where {T <: AbstractArray}
  return x[[axes(x, i) for i in 1:ndims(x)-1]..., l]
end

"""
returns: x[[axes(x, i) for i in 1:ndims(x)-1]..., end]
"""
function tail(x::T) where {T <: AbstractArray}
  return x[[axes(x, i) for i in 1:ndims(x)-1]..., end]
end

"""
logsumexp
"""
function logsumexp(logx::T; dims::Integer=1) where {T <: AbstractArray}
  d = dims < 0 ? ndims(logx) + 1 + dims : dims
  mx = maximum(logx, dims=d)
  return mx .+ log.(sum(exp.(logx .- mx), dims=d))
end

"""
return: dropdims(logsumexp(logx, dims=dims), dims=dims)
"""
function logsumexpdd(logx::T; dims::Integer=1) where {T <: AbstractArray}
  d = dims < 0 ? ndims(logx) + 1 + dims : dims
  return dropdims(logsumexp(logx, dims=d), dims=d)
end

"""
return: dropdims(sum(x, dims=dims), dims=dims)
"""
function sumdd(x::T; dims::Integer=1) where {T <: AbstractArray}
  d = dims < 0 ? ndims(x) + 1 + dims : dims
  return dropdims(sum(x, dims=d), dims=d)
end

"""
return: dropdims(prod(x, dims=dims), dims=dims)
"""
function proddd(x::T; dims::Integer=1) where {T <: AbstractArray}
  d = dims < 0 ? ndims(x) + 1 + dims : dims
  return dropdims(prod(x, dims=d), dims=d)
end

function lpdf_normal(x::X, m::M, s::S) where {X <: Real, M <: Real, S<:Real}
  z = (x - m) / s
  return oftype(z, -0.5 * log(2*pi) - z^2 * 0.5 - log(s))
end

"""
log pdf of gaussian mixture model
"""
function lpdf_gmm(x::TX, m::TM, s::TS, w::TW; dims::Integer, dropdim::Bool=false) where {TX, TW, TM, TS}
  if dropdim
    return logsumexpdd(log.(w) .+ lpdf_normal.(x, m, s), dims=dims)
  else
    return logsumexp(log.(w) .+ lpdf_normal.(x, m, s), dims=dims)
  end
end

"""
cat(a, b, dims=ndims(A) + 1)
"""
function stack(a::A, b::B) where {A, B}
  @assert size(a) == size(b)
  return cat(a, b, dims=ndims(a) + 1)
end
