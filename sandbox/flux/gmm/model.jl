using Flux, Flux.Tracker
using Distributions

struct VP
  m
  s
  w

  VP(K::Integer) = new(param(randn(K, 2)), param(randn(K, 2)), param(randn(K - 1, 2)))
end

function logsumexp(logx::T) where T
  mx = maximum(logx)
  return mx + log(sum(exp.(logx .- mx)))
end

function lpdf_logx(dist::D, logx::T) where {D, T}
  return logpdf(dist, exp(logx)) + logx
end

function softmax_safe(x::T) where T
  return exp.(x .- logsumexp(x))
end

function lpdf_real_simplex(dist::D, real_simplex::T) where {D, T}
  return # TODO
end
