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

function Priors(K::Int, L::Dict{Bool, Int}; T::Type=Float64, use_stickbreak::Bool=false)
  v_prior = use_stickbreak ? a -> Beta(a, T(1)) : a -> Beta(a / K, T(1))
  return Priors(Gamma(T.((1, 1))...), # delta0
                Gamma(T.((1, 1))...), # delta1
                LogNormal(T.((-1, 1))...), # sig2
                Dirichlet(1 ./ ones(T, K)), # W
                Dirichlet(1 ./ ones(T, L[0])), # eta0
                Dirichlet(1 ./ ones(T, L[1])), # eta1
                v_prior, # v
                Uniform(T.((0, 1))...), # H
                Gamma(T.((0.1, 0.1))...), # alpha
                Beta(T.((1, 99))...)) # eps
end
