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
  beta::Matrix{Float64} # beta_dims x I, beta[:, i] refers to the beta's for sample i
  K::Int
  L::Dict{Int, Int}
  # For repulsive Z
  probFlip_Z::Float64
  similarity_Z::Function
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
                          tau0::Float64=0.0, tau1::Float64=0.0,
                          probFlip_Z::Float64=1.0 / (data.J * K),
                          similarity_Z::Function=sim_fn_abs(0))
  alpha_prior = Gamma(3.0, 0.5)
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
  mus_prior[0] = TruncatedNormal(mean(y_neg), tau0, -10,  0)
  mus_prior[1] = TruncatedNormal(mean(y_pos), tau1,   0, 10)
  W_prior = Dirichlet(K, 1 / K)
  eta_prior = Dict(z => Dirichlet(L[z], 1 / L[z]) for z in 0:1)
  sig2_prior = InverseGamma(3.0, 2 / 3)

  # TODO: use empirical bayes to find these priors
  y_negs = [filter(y_i -> !isnan(y_i) && y_i < 0, vec(data.y[i])) for i in 1:data.I]
  beta = hcat([gen_beta_est(y_negs[i], yQuantiles, pBounds) for i in 1:data.I]...)

  return Constants(alpha_prior=alpha_prior, mus_prior=mus_prior, W_prior=W_prior,
                   eta_prior=eta_prior, sig2_prior=sig2_prior,
                   beta=beta, K=K, L=L,
                   probFlip_Z=probFlip_Z, similarity_Z=similarity_Z)
end

function priorMu(z::Int, l::Int, s::State, c::Constants)
  L = c.L

  if l == 1
    lower, upper = minimum(c.mus_prior[z]), s.mus[z][l+1]
  elseif l == L[z]
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
