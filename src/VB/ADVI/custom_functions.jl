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

function lpdf_dirichlet(p::AbstractArray, a::AbstractArray)
  if size(a) != size(p)
    a = reshape(a, (one.(size(p)[1:end-1])..., length(a)))
  end
  K = ndims(p)
  return Tracker.lgamma.(sumdd(a, dims=K)) .- sumdd(Tracker.lgamma.(a), dims=K) .+ sumdd((a .- 1) .* log.(p), dims=K)
end

function lpdf_beta(p::P, a::A, b::B) where {P <: Real, A <: Real, B <: Real}
  return Tracker.lgamma(a + b) - Tracker.lgamma(a) - Tracker.lgamma(b) + (a-1)*log(p) + (b-1)*log1p(-p)
end

function lpdf_gamma(x::X, shape::A, scale::B) where {X <: Real, A <: Real, B <: Real}
  return -(Tracker.lgamma(shape) + shape*log(scale)) + (shape - 1)*log(x) - x / scale
end

function lpdf_lognormal(x::X, m::A, s::B) where {X <: Real, A <: Real, B <: Real}
  z = (log(x) - m) / s
  return X(-log(x + s) - 0.5*log(2*pi) - z^2 / 2)
end

function lpdf_uniform(x::X, a::A, b::B) where {X <: Real, A <: Real, B <: Real}
  @assert b > a
  if a <= x <= b
    return -log(b - a)
  else
    return X(-Inf)
  end
end

compute_lpdf(dist::Dirichlet, x::X) where X = lpdf_dirichlet(x, dist.alpha)
compute_lpdf(dist::Normal, x::X) where X = lpdf_normal.(x, dist.μ, dist.σ)
compute_lpdf(dist::Beta, x::X) where X = lpdf_beta.(x, dist.α, dist.β)
compute_lpdf(dist::Gamma, x::X) where X = lpdf_gamma.(x, dist.α, dist.θ)
compute_lpdf(dist::LogNormal, x::X) where X = lpdf_lognormal.(x, dist.μ, dist.σ)
compute_lpdf(dist::Uniform, x::X) where X = lpdf_uniform.(x, dist.a, dist.b)
