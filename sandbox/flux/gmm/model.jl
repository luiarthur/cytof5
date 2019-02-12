using Flux, Flux.Tracker
using Distributions
import LinearAlgebra.logdet
import SpecialFunctions.lgamma

TA = typeof(param(randn(3, 2)))

struct VP
  m::TA
  log_s::TA
  real_w::TA

  VP(K::Integer) = new(param(randn(K, 2)), param(randn(K, 2)), param(zeros(K - 1, 2)))
end

function lpdf_vp(p::S, x::T) where {S, T}
  return sum(logpdf.(Normal.(p[:, 1], exp.(p[:, 2])), x))
end

function rsample(p::T) where T
  return p[:, 1] .+ randn(size(p, 1)) .* exp.(p[:, 2])
end

function logsumexp(logx::T) where T
  mx = maximum(logx)
  return mx + log(sum(exp.(logx .- mx)))
end

function logsumexp0p(logx::T) where T
  return log1p(sum(exp.(logx)))
end

function lpdf_logx(dist::D, logx::T) where {D, T}
  return logpdf(dist, exp(logx)) + logx
end

function softmax_safe(x::T) where T
  logsumexp_x = logsumexp(x)
  return exp.(x .- logsumexp_x)
end

"""
Transform a real vector to a simplex. The sum of the resulting simplex
is less than 1, and one minus that sum is the probability of the first 
element in the K+1 dimensional simplex.
"""
function softmax_fullrank(x::T; complete::Bool=true) where T
  logsumexp0p_x = logsumexp0p(x)
  p_reduced = exp.(x .- logsumexp0p_x)
  if complete
    K = length(p_reduced) + 1
    g =  vcat(1 .- sum(p_reduced, dims=1), p_reduced)
    return g
  else
    return p_reduced
  end
end

function lpdf_Dirichlet_fullrank(a::A, p::T) where {A, T}
  out = lgamma(sum(a)) - sum(lgamma.(a)) + sum((a[2:end] .- 1.0) .* log.(p))
  out += (a[1] - 1.0) * log1p(-sum(p))
  return out
end

# TODO: Test
function lpdf_real_simplex(alpha::S, real_simplex::T) where {S, T}
  dim = length(real_simplex)
  p = softmax_fullrank(real_simplex, complete=false)
  jacobian = [i == j ? p[i] * (1 - p[i]) : -p[i] * p[j] for i in 1:dim, j in 1:dim]
  return lpdf_Dirichlet_fullrank(alpha, p) + logdet(jacobian)
end


# loglike
function loglike(y::S, w::T, m::T, s::T) where {S, T}
  N = length(y)
  # println(s)
  out = 0.0
  for i in 1:N
    out += sum(logsumexp(log.(w .+ 1e-6) .+ logpdf.(Normal.(m, s), y[i])))
  end
  return out
end

# log_p
function log_p(real_w::T, m::T, log_s::T) where T
  K = length(m)
  log_p_m = sum(logpdf.(Normal(0, 1), m))
  log_p_log_s = sum(lpdf_logx.(LogNormal(-1, .1), log_s))
  log_p_real_w = lpdf_real_simplex(ones(K) * 10, real_w)
  return log_p_m + log_p_log_s + log_p_real_w
end

# log_q
function log_q(real_w::T, m::T, log_s::T, vp::VP) where T
  log_q_m = lpdf_vp(vp.m, m)
  log_q_log_s = lpdf_vp(vp.log_s, log_s)
  log_q_real_w = lpdf_vp(vp.real_w, real_w)
  return log_q_m + log_q_log_s + log_q_real_w
end

# ELBO
function elbo(y::T, vp::VP) where {T}
  m = rsample(vp.m)

  log_s = rsample(vp.log_s)
  s = exp.(log_s)

  real_w = rsample(vp.real_w)
  w = softmax_fullrank(real_w)

  return loglike(y, w, m, s) + log_p(real_w, m, log_s) - log_q(real_w, m, log_s, vp)
end

