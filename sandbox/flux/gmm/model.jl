using Flux, Flux.Tracker
using Distributions
import LinearAlgebra.logdet

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

"""
Transform a real vector of size K to a simplex of size K + 1.  This is for
identifiability. The first dimension of the simplex is 0 in the real vector
(hence it is one dimension less).
"""
function softmax_fullrank(x::T) where T
  x_aug = [0.0; x]
  return softmax_safe(x_aug)
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
        jacobian[i, j] = p[i+1] * (1 - p[i+1]) # TODO
      else
        jacobian[i, j] = jacobian[j, i] = -p[i+1] * p[j+1] # TODO
      end
    end
  end
  return logpdf(dist, p) + logdet(jacobian)
end
