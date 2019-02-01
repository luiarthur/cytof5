function arrMatTo3dArr(x)
  @assert all(length.(x) .== length(x[1,1]))
  K = length(x[1,1])
  return [x[i,j][k] for i in 1:size(x,1), j in 1:size(x,2), k in 1:K]
end

@namedargs mutable struct Constants
  alpha_prior::Gamma # alpha ~ Gamma(shape, scale)
  mus_prior::Dict{Int, Truncated{Normal{Float64}, Continuous}} # mu*[z,l] ~ TN(mean,sd)
  W_prior::Dirichlet # W_i ~ Dir_K(d)
  eta_prior::Dict{Int, Dirichlet{Float64}} # eta_zij ~ Dir_Lz(a)
  sig2_prior::InverseGamma # sig2_i ~ IG(shape, scale)
  sig2_range::Vector{Float64} # lower and upper bound for sig2
  beta::Matrix{Float64} # beta_dims x I, beta[:, i] refers to the beta's for sample i
  eps_prior::Vector{Beta{Float64}} # I-dim
  K::Int
  L::Dict{Int, Int}
  # For repulsive Z
  probFlip_Z::Float64
  similarity_Z::Function
  noisyDist::ContinuousDistribution
  y_grid::Vector{Float64}
  iota_prior::Gamma
end

"""
yi: y[i] = N[i] x J matrix
pBounds = the bounds for probability of missing to compute the missing mechanism.
yQuantiles = the quantiles to compute the lower and upper bounds for y for the missing mechanism.
"""
function gen_beta_est(yi, yQuantiles, pBounds)
  yiNeg = yi[ (isnan.(yi) .== false) .& (yi .< 0) ]
  yBounds = quantile(yiNeg, yQuantiles)

  return solveBeta(yBounds, pBounds)
end

"""
Genearte default values for constants
"""
function defaultConstants(data::Data, K::Int, L::Dict{Int, Int};
                          pBounds=[.01, .8, .05], yQuantiles=[0.01, .1, .25],
                          sig2_prior=InverseGamma(3.0, 2 / 3),
                          sig2_range=[0.0, Inf],
                          mus0_range=[-20, 0.0],
                          mus1_range=[0.0, 20],
                          alpha_prior = Gamma(3.0, 0.5),
                          iota_prior=Gamma(5.0, 0.1), # shape and scale
                          tau0::Float64=0.0, tau1::Float64=0.0,
                          probFlip_Z::Float64=1.0 / (data.J * K),
                          noisyDist::ContinuousDistribution=Cauchy(),
                          # noisyDist::ContinuousDistribution=Normal(0.0, sqrt(10.0)),
                          y_grid::Vector{Float64}=collect(range(-10, stop=4, length=100)),
                          similarity_Z::Function=sim_fn_abs(0))
  # Assert range of sig2 is positive
  @assert 0 <= sig2_range[1] < sig2_range[2]

  mus_prior = Dict{Int, Truncated{Normal{Float64}, Continuous}}()
  vec_y = vcat(vec.(data.y)...)
  y_neg = filter(y_inj -> !isnan(y_inj) && y_inj < 0, vec_y)
  y_pos = filter(y_inj -> !isnan(y_inj) && y_inj > 0, vec_y)
  if tau0 <= 0
    tau0 = std(y_neg)
  end
  if tau1 <= 0
    tau1 = std(y_pos)
  end
  mus_prior[0] = TruncatedNormal(mean(y_neg), tau0, mus0_range[1], mus0_range[2])
  mus_prior[1] = TruncatedNormal(mean(y_pos), tau1, mus1_range[1], mus1_range[2])

  W_prior = Dirichlet(K, 1 / K)
  eta_prior = Dict(z => Dirichlet(L[z], 1 / L[z]) for z in 0:1)
  eps_prior = [Beta(5.0, 95.0) for i in 1:data.I]

  # TODO: use empirical bayes to find these priors
  y_negs = [filter(y_i -> !isnan(y_i) && y_i < 0, vec(data.y[i])) for i in 1:data.I]
  beta = hcat([gen_beta_est(y_negs[i], yQuantiles, pBounds) for i in 1:data.I]...)

  return Constants(alpha_prior=alpha_prior, mus_prior=mus_prior, W_prior=W_prior,
                   eta_prior=eta_prior,
                   sig2_prior=sig2_prior, sig2_range=sig2_range,
                   beta=beta, K=K, L=L,
                   probFlip_Z=probFlip_Z, similarity_Z=similarity_Z,
                   noisyDist=noisyDist, eps_prior=eps_prior, y_grid=y_grid,
                   iota_prior=iota_prior)
end

function priorMu(z::Int, l::Int, s::State, c::Constants)
  L = c.L

  if l == 1
    if z == 1 # z==1 & l==1 => for mu[1][1]
      lower, upper = s.iota, s.mus[z][l+1]
    else # z==0 & l==1 => for mu[0][1]
      lower, upper = minimum(c.mus_prior[z]), s.mus[z][l+1]
    end
  elseif l == L[z]
    if z == 1 # z == 1 & l == L[1] => for mu[1][end]
      lower, upper = s.mus[z][l-1], maximum(c.mus_prior[z])
    else # z == 0 & l == L[0] => for mu[0][end]
      lower, upper = s.mus[z][l-1], -s.iota
    end
  else
    lower, upper = s.mus[z][l-1], s.mus[z][l+1]
  end

  # Note that priorM and priorS are NOT the prior mean and std. They are PARAMETERS in 
  # the truncated normal!
  (priorM, priorS, _, _) = params(c.mus_prior[z])
  # println("z: $z, l: $l | pm: $priorM, ps: $priorS, low: $lower, upp: $upper")
  #priorMean = mean(c.mus_prior[z])
  #priorSd = std(c.mus_prior[z])
  return TruncatedNormal(priorM, priorS, lower, upper)
end

function printConstants(c::Constants, preprintln::Bool=true)
  if preprintln
    println("Constants:")
  end

  for fname in fieldnames(typeof(c))
    x = getfield(c, fname)
    T = typeof(x)
    if T <: Number
      println("$fname: $x")
    elseif T <: Vector
      N = length(x)
      for i in 1:N
        println("$(fname)[$i]: $(x[i])")
      end
    elseif T <: Dict
      for (k, v) in x
        println("$(fname)[$k]: $v")
      end
    else
      println("$fname: $x")
    end
  end
end
