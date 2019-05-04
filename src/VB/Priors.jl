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
  v_prior = use_stickbreak ? a -> Beta(a, 1) : a -> Beta(a / K, 1)
  return Priors(Gamma(1, 1),# delta0
                Gamma(1, 1), # delta1
                LogNormal(-1, 1), # sig2
                Dirichlet(1 ./ ones(K)), # W
                Dirichlet(1 ./ ones(L[0])), # eta0
                Dirichlet(1 ./ ones(L[1])), # eta1
                v_prior, # v
                Uniform(0, 1), # H
                Gamma(0.1, 0.1), # alpha
                Beta(1, 99)) # eps
end
