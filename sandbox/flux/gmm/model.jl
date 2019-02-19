using Flux, Flux.Tracker
using Distributions
import LinearAlgebra.logabsdet
import SpecialFunctions.lgamma

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


function logsumexp0p(logx)
  return log1p(sum(exp.(logx)))
end

function lpdf_logx(dist, logx)
  return logpdf(dist, exp(logx)) + logx
end

function softmax_safe(x)
  logsumexp_x = logsumexp(x)
  return exp.(x .- logsumexp_x)
end

"""
Transform a real vector to a simplex. The sum of the resulting simplex
is less than 1, and one minus that sum is the probability of the first 
element in the K+1 dimensional simplex.
"""
function softmax_fullrank(x; complete::Bool=true)
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

function lpdf_Dirichlet_fullrank(a, p)
  out = lgamma(sum(a)) - sum(lgamma.(a)) + sum((a[2:end] .- 1.0) .* log.(p))
  out += (a[1] - 1.0) * log1p(-sum(p))
  return out
end

# TODO: Test
function lpdf_real_simplex(alpha, real_simplex)
  dim = length(real_simplex)
  p = softmax_fullrank(real_simplex, complete=false)
  jacobian = [i == j ? p[i] * (1 - p[i]) : -p[i] * p[j] for i in 1:dim, j in 1:dim]
  return lpdf_Dirichlet_fullrank(alpha, p) + logabsdet(jacobian)[1]
end

# loglike
function loglike(y, w, m, s)
  N = length(y)

  # NOTE: This is slow because of how tracker is implemented.
  #       Tensor expressions are still needed.
  # out = 0.0
  # for i in 1:N
  #   out += sum(logsumexp(log.(w) .+ logpdf.(Normal.(m, s), y[i]), dims=1))
  # end
  # return out

  # NOTE: Preferred Implementation
  K = length(m)
  log_w = reshape(log.(w), 1, K)
  lpdf = logpdf.(Normal.(reshape(m, 1, K), reshape(s, 1, K)), y)
  return sum(logsumexp(log_w .+ lpdf, dims=2))

  # NOTE: Tracker can't keep track of reshapes of Distribution objects
  #       i.e. REPLACING THE PREVIOUS TWO LINES WITH THE FOLLOWING WILL NOT
  #       WORK!!!
  # lpdf = logpdf.(reshape(Normal.(m, s), 1, K), reshape(y, N, 1))
  # return sum(logsumexp(log_w .+ lpdf, dims=2))
end

# log_p
function log_p(real_w, m, log_s)
  K = length(m)
  log_p_m = sum(logpdf.(Normal(0, 1), m))
  log_p_log_s = sum(lpdf_logx.(LogNormal(0, 1), log_s))
  log_p_real_w = lpdf_real_simplex(ones(K) / K, real_w)
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
  w = softmax_fullrank(real_w)

  return loglike(y, w, m, s) + log_p(real_w, m, log_s) - log_q(real_w, m, log_s, vp)
end

