function arrMatTo3dArr(x)
  @assert all(length.(x) .== length(x[1,1]))
  K = length(x[1,1])
  return [x[i,j][k] for i in 1:size(x,1), j in 1:size(x,2), k in 1:K]
end

mutable struct Constants
  alpha_prior::Gamma # alpha ~ Gamma(shape, scale)
  mus_prior::Dict{Int, Truncated{Normal{Float64}, Continuous}} # mu*[z,l] ~ TN(mean,sd)
  W_prior::Dirichlet # W_i ~ Dir_K(d)
  eta_prior::Dirichlet # eta_zij ~ Dir_L(a)
  sig2_prior::InverseGamma # sig2_i ~ IG(shape, scale)
  b0_prior::Vector{Normal} # b0 ~ Normal(mean, sd)
  b1_prior::Vector{Gamma} # b1 ~ Gamma(shape, scale) (positive)
  #b1_prior::Vector{Uniform} # b1 ~ Unif(a, b) (positive)
  K::Int
  L::Int
  # For repulsive Z
  probFlip_Z::Float64
  similarity_Z::Function
end

"""
yi: y[i] = N[i] x J matrix
pBounds = the bounds for probability of missing to compute the missing mechanism.
yQuantiles = the quantiles to compute the lower and upper bounds for y for the missing mechanism.
"""
function genBPrior(yi, pBounds, yQuantiles)
  @assert pBounds[1] > pBounds[2]
  @assert yQuantiles[1] < yQuantiles[2]

  yiNeg = yi[ (isnan.(yi) .== false) .& (yi .< 0) ]
  yLower = quantile(yiNeg, yQuantiles[1])
  yUpper = quantile(yiNeg, yQuantiles[2])
  yBounds = (yLower, yUpper)

  return solveB(yBounds, pBounds)
end

"""
b0PriorSd: bigger -> more uncertainty.
b1PriorScale: bigger -> more uncertainty. prior scale is the empirical mean / scale. So prior mean is empirical mean.
"""
function defaultConstants(data::Data, K::Int, L::Int; pBounds=(.9, .01), yQuantiles=(.01, .10),
                          b0PriorSd::Number=1.0, b1PriorScale::Number=1/10,
                          probFlip_Z::Float64=1.0 / (data.J * K))
  alpha_prior = Gamma(3.0, 0.5)
  mus_prior = Dict{Int, Truncated{Normal{Float64}, Continuous}}()
  vec_y = vcat(vec.(data.y)...)
  y_neg = filter(y_inj -> !isnan(y_inj) && y_inj < 0, vec_y)
  y_pos = filter(y_inj -> !isnan(y_inj) && y_inj > 0, vec_y)
  mus_prior[0] = TruncatedNormal(mean(y_neg), std(y_neg), -10, 0)
  mus_prior[1] = TruncatedNormal(mean(y_pos), std(y_pos), 0, 10)
  W_prior = Dirichlet(K, 1.0/K)
  eta_prior = Dirichlet(L, 1.0 / L)
  sig2_prior = InverseGamma(3.0, 2.0)

  # TODO: use empirical bayes to find these priors
  #b0_prior = [ Normal(-9.2, 1.0) for i in 1:data.I ]
  #b1_prior = [ Gamma(2.0, 1.0) for i in 1:data.I ]
  b0_prior = [ Normal(genBPrior(vec(data.y[i]), pBounds, yQuantiles)[1], b0PriorSd) for i in 1:data.I ]
  b1_prior = [ Gamma(genBPrior(vec(data.y[i]), pBounds, yQuantiles)[2]/b1PriorScale, b1PriorScale) for i in 1:data.I ]

  #b0_prior = Uniform(1.0, 3.0)
  #b1_prior = Uniform(0.0, 20.0)

  return Constants(alpha_prior, mus_prior, W_prior, eta_prior, sig2_prior,
                   b0_prior, b1_prior, K, L, probFlip_Z, similarity_default)
end

# TODO: Test
function priorMu(z::Int, l::Int, s::State, c::Constants)
  L = c.L

  if l == 1
    lower, upper = minimum(c.mus_prior[z]), s.mus[z][l+1]
  elseif l == L
    lower, upper = s.mus[z][l-1], maximum(c.mus_prior[z])
  else
    lower, upper = s.mus[z][l-1], s.mus[z][l+1]
  end

  # Note that priorM and priorS are NOT the prior mean and std. They are PARAMETERS in 
  # the truncated normal!
  (priorM, priorS, _, _) = params(c.mus_prior[z])
  #priorMean = mean(c.mus_prior[z])
  #priorSd = std(c.mus_prior[z])
  return TruncatedNormal(priorM, priorS, lower, upper)
end

# TODO
function genInitialState(c::Constants, d::Data)
  J = d.J
  K = c.K
  L = c.L
  I = d.I
  N = d.N

  yVecObserved = [ filter(yinj -> !isnan(yinj), vec(d.y[i])) for i in 1:I ]

  y_imputed = begin
    local out = [zeros(Float64, N[i], J) for i in 1:I]
    y_lower, y_upper = quantile.(c.mus_prior[0], [0, .1])
    for i in 1:I
      for n in 1:N[i]
        for j in 1:J
          if isnan(d.y[i][n, j])
            # out[i][n, j] = rand(c.mus_prior[0])
            out[i][n, j] = rand(Uniform(y_lower, y_upper))
          else
            out[i][n, j] = d.y[i][n, j]
          end
        end
      end
    end

    out
  end

  alpha = rand(c.alpha_prior)
  v = rand(Beta(alpha / c.K, 1), K)
  Z = [ rand(Bernoulli(v[k])) for j in 1:J, k in 1:K ]
  mus = Dict([z => sort(rand(c.mus_prior[z], L)) for z in 0:1])
  sig2 = [rand(c.sig2_prior) for i in 1:I]
  b0 = [ rand(c.b0_prior[i]) for i in 1:I ]
  b1 = [ rand(c.b1_prior[i]) for i in 1:I ]
  W = Matrix{Float64}(hcat([ rand(c.W_prior) for i in 1:I ]...)')
  lam = [ rand(Categorical(W[i,:]), N[i]) for i in 1:I ]
  eta = begin
    function gen(z)
      arrMatTo3dArr([ rand(c.eta_prior) for i in 1:I, j in 1:J ])
    end
    Dict([z => gen(z) for z in 0:1])
  end
  gam = [[rand(Categorical(eta[Z[j, lam[i][n]]][i, j, :])) for n in 1:N[i], j in 1:J] for i in 1:I]

  #@assert typeof(Z) <: Matrix{Int}
  #@assert typeof(mus) <: Dict{Tuple{Int,Int}, Float64}
  #@assert typeof(alpha) == Float64
  #@assert typeof(v) == Vector{Float64}
  #@assert typeof(W) == Matrix{Float64}
  #@assert typeof(sig2) == Vector{Float64}
  #@assert typeof(eta) == Dict{Int, Cube{Float64}}
  #@assert typeof(lam) == Vector{Vector{Int}}
  #@assert typeof(b0) == Vector{Float64}
  #@assert typeof(b1) == Vector{Float64}
  #@assert typeof(gam) == Vector{Matrix{Int}}
  #@assert typeof(y_imputed) == Vector{Matrix{Float64}}

  return State(Z, mus, alpha, v, W, sig2, eta, lam, gam, y_imputed, b0, b1)
end
