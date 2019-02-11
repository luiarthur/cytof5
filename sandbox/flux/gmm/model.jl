using Flux, Flux.Tracker
using Distributions
import LinearAlgebra.logdet
import SpecialFunctions.lgamma

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

function logsumexp0p(logx::T) where T
  mx = max(maximum(logx), 0.0)
  return mx + log(exp(-mx) + sum(exp.(logx .- mx)))
end

# Test
# for i in 1:1000
#   x = randn(100)
#   @assert all(abs(logsumexp([0.0; x]) .- logsumexp0p(x)) < 1e-8)
# end

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
function softmax_fullrank(x::T) where T
  logsumexp0p_x = logsumexp0p(x)
  return exp.(x .- logsumexp0p_x)
end

function lpdf_Dirichlet_fullrank(a::A, p::T) where {A, T}
  out = lgamma(sum(a)) - sum(lgamma.(a)) + sum((a[2:end] .- 1.0) .* log.(p))
  out += (a[1] - 1.0) * log1p(-sum(p))
  return out
end

# TODO: Test
function lpdf_real_simplex(dist::D, real_simplex::T) where {D, T}
  dim = length(real_simplex)
  p = softmax_fullrank(real_simplex) # dimensions: dim + 1
  xtype = typeof(real_simplex[1])
  jacobian = zeros(xtype, dim, dim)
  for i in 1:dim
    for j in 1:dim
      if i == j
        jacobian[i, j] = p[i] * (1 - p[i]) # TODO
      else
        jacobian[i, j] = jacobian[j, i] = -p[i] * p[j] # TODO
      end
    end
  end
  return lpdf_Dirichlet_fullrank(dist.alpha, p) + logdet(jacobian)
end


# Test
K = 30
a = collect(1:K)

x = randn(K - 1)
y = param(x)

@time for i in 1:100
  A = lpdf_real_simplex(Dirichlet(a), x)
  B = lpdf_real_simplex(Dirichlet(a), y)
  @assert abs(A - B) < 1e-6

  C = lpdf_Dirichlet_fullrank(a, softmax_fullrank(x))
  D = logpdf(Dirichlet(a), softmax([0.; x]))
  @assert abs(C - D) < 1e-6

  # FIXME: prepend not available here
  # p = softmax_fullrank(y)
  # prepend!(p, 1 - sum(p))
  # @assert abs(1 - sum(p)) < 1e-6
end
# Test
