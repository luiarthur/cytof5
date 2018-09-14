struct Constants
  alpha_prior::Gamma # alpha ~ Gamma(shape, scale)
  mus_prior::Dict{Int, Truncated{Normal{Float64}, Continuous}} # mu*[z,l] ~ TN(mean,sd)
  W_prior::Dirichlet # W_i ~ Dir_K(d)
  sig2_prior::InverseGamma # sig2_i ~ IG(shape, scale)
  b0_prior::Normal # b0 ~ Normal(mean, sd)
  b1_prior::Uniform # b1 ~ Unif(a, b) (positive)
  K::Int
  L::Int
end

function defaultConstants(data::Data, K::Int, L::Int)
  alpha_prior = Gamma(3.0, 0.5)
  mus_prior = Dict{Int, Truncated{Normal{Float64}, Continuous}}()
  vec_y = vcat(vec.(data.y)...)
  y_neg = filter(y_inj -> !ismissing(y_inj) && y_inj < 0, vec_y)
  y_pos = filter(y_inj -> !ismissing(y_inj) && y_inj > 0, vec_y)
  mus_prior[0] = TruncatedNormal(mean(y_neg), std(y_neg), -30, 0)
  mus_prior[1] = TruncatedNormal(mean(y_pos), std(y_pos), 0, 30)
  W_prior = Dirichlet(K, 1.0/K)
  sig2_prior = InverseGamma(3.0, 2.0)
  b0_prior = Normal(0.0, 10.0)
  b1_prior = Uniform(0.0, 50.0)

  Constants(alpha_prior, mus_prior, W_prior, sig2_prior, b0_prior, b1_prior, K, L)
end

# TODO: Test
function priorMu(s::State, c::Constants, z::Int, l::Int)
  L = c.L

  if l == 1
    lower, upper = minimum(c.mus_prior[z]), s.mus[z][2]
  elseif l == L
    lower, upper = s.mus[z][L-1], maximum(c.mus_prior[z])
  else
    lower, upper = s.mus[z][l-1], s.mus[z][l+1]
  end

  return TruncatedNormal(mean(c.mus_prior[z]), sd, lower, upper)
end

# TODO
function genInitialState(c::Constants, d::Data)
  J = d.J
  K = c.K

  alpha = rand(c.alpha_prior)
  v = rand(Beta(alpha / c.K, 1), K)
  Z = [ rand(Bernoulli(v[k])) for j in 1:J, k in 1:K]
  #mus = Dict([ z =>  for z in 0:1 ])

  #State(Z, mus, alpha, v, W, sig2, eta, lam, gam, y_imputed, b0, b1)
  return 0
end
