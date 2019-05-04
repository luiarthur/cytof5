mutable struct Priors
  delta0
  delta1
  sig2
  W
  eta0
  eta1
  v
  H
  alpha
  eps
end

# Gamma(shape, scale)

function Priors(K::Int, L::Dict{Bool, Int}; use_stickbreak::Bool=false)
  v_prior = use_stickbreak ? a -> Beta(a, oftype(a, 1)) : a -> Beta(a / K, oftype(a, 1))
  return Priors(Gamma(1, 1),# delta0
                Gamma(1, 1), # delta1
                LogNormal(-1, .1), # sig2
                Dirichlet(ones(K) / K), # W
                Dirichlet(ones(L[0]) / L[0]), # eta0
                Dirichlet(ones(L[1]) / L[1]), # eta1
                v_prior, # v
                Uniform(0, 1), # H
                Gamma(0.1, 0.1), # alpha
                Beta(1, 99)) # eps
end
