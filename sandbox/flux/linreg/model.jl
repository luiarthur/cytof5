using Flux, Flux.Tracker
using Distributions

struct VP
  b0
  b1
  log_sig

  VP() = new(param(randn(2)), param(randn(2)), param(randn(2)))
end

function lpdf_logx(dist, logx)
  return logpdf(dist, exp(logx)) + logx
end

function loglike(y::Vector{S}, x::Vector{S}, b0::T, b1::T, sig::T, N::Int) where {T, S}
  return sum(logpdf.(Normal.(b0 .+ b1 .* x, sig), y)) * (N / length(y))
end

function log_p(b0::T, b1::T, log_sig::T) where T
  return logpdf(Normal(0, 1), b0) + logpdf(Normal(0, 1), b1) + lpdf_logx(Gamma(1, 1), log_sig)
end

lpdf_vp(p, x) = logpdf(Normal(p[1], exp(p[2])), x)
function log_q(b0::T, b1::T, log_sig::T, vp::VP) where T
  return lpdf_vp(vp.b0, b0) + lpdf_vp(vp.b1, b1) + lpdf_vp(vp.log_sig, log_sig)
end

rsample(p) = randn() * exp(p[2]) + p[1]
function elbo(y::Vector{T}, x::Vector{T}, vp, N::Int) where {T}
  b0 = rsample(vp.b0)
  b1 = rsample(vp.b1)
  log_sig = rsample(vp.log_sig)
  sig = exp(log_sig)

  return loglike(y, x, b0, b1, sig, N) + log_p(b0, b1, log_sig) - log_q(b0, b1, log_sig, vp)
end

