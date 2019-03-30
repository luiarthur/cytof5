using Flux, Flux.Tracker
using Distributions
import LinearAlgebra.logabsdet
import SpecialFunctions.lgamma
include("StickBreak.jl"); SB = StickBreak

struct VP
  m::TrackedArray
  log_s::TrackedArray
  real_w::TrackedArray

  VP(K::Integer) = new(param(randn(K, 2)), param(randn(K, 2)), param(zeros(K - 1, 2)))
end

function lpdf_vp(p, x)
  return sum(logpdf.(Normal.(p[:, 1], exp.(p[:, 2])), x))
end

function rsample(p)
  return p[:, 1] .+ randn(size(p, 1)) .* exp.(p[:, 2])
end

function logsumexp(logx)
  mx = maximum(logx)
  return mx + log(sum(exp.(logx .- mx)))
end

function logsumexp(logx; dims::Integer)
  mx = maximum(logx, dims=dims)
  return mx .+ log.(sum(exp.(logx .- mx), dims=dims))
end

function lpdf_logx(dist, logx)
  return logpdf(dist, exp(logx)) + logx
end

function lpdf_real_simplex(alpha, x, p)
  return logpdf(Dirichlet(alpha), p) + SB.logabsdetJ(x, p)
end

# loglike
function loglike(y, w, m, s)
  N = length(y)
  K = length(m)
  log_w = reshape(log.(w), 1, K)
  lpdf = logpdf.(Normal.(reshape(m, 1, K), reshape(s, 1, K)), y)

  return sum(logsumexp(log_w .+ lpdf, dims=2))
end

# log_p
function log_p(real_w, w, m, log_s)
  K = length(m)
  log_p_m = sum(logpdf.(Normal(0, 10), m))
  log_p_log_s = sum(lpdf_logx.(LogNormal(0, 1), log_s))
  log_p_real_w = lpdf_real_simplex(ones(K) / K, real_w, w)
  return log_p_m + log_p_log_s + log_p_real_w
end

# log_q
function log_q(real_w, m, log_s, vp::VP)
  log_q_m = lpdf_vp(vp.m, m)
  log_q_log_s = lpdf_vp(vp.log_s, log_s)
  log_q_real_w = lpdf_vp(vp.real_w, real_w)
  return log_q_m + log_q_log_s + log_q_real_w
end

# ELBO
function elbo(y, vp::VP)
  m = rsample(vp.m)

  log_s = rsample(vp.log_s)
  s = exp.(log_s)

  real_w = rsample(vp.real_w)
  w = SB.transform(sigmoid.(real_w) * 20)

  return loglike(y, w, m, s) + log_p(real_w, w, m, log_s) - log_q(real_w, m, log_s, vp)
end

