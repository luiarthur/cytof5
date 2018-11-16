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
                          pBounds=[.05, .8, .05], yQuantiles=[0, .1, .25],
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

# TODO
function genInitialState(c::Constants, d::Data)
  J = d.J
  K = c.K
  L = c.L
  I = d.I
  N = d.N

  vec_y = vcat(vec.(d.y)...)
  y_neg = filter(y_inj -> !isnan(y_inj) && y_inj < 0, vec_y)

  y_imputed = begin
    local out = [zeros(Float64, N[i], J) for i in 1:I]
    # y_lower, y_upper = quantile.(c.mus_prior[0], [0, .1])
    y_lower, y_upper = quantile(y_neg, [0, .1])
    for i in 1:I
      for n in 1:N[i]
        for j in 1:J
          if isnan(d.y[i][n, j])
            # out[i][n, j] = rand(c.mus_prior[0])
            out[i][n, j] = rand(Uniform(y_lower, y_upper))
          else
            out[i][n, j] = d.y[i][n, j]
          end
          @assert !isnan(out[i][n, j])
        end
      end
    end

    out
  end

  alpha = rand(c.alpha_prior)
  v = rand(Beta(alpha / c.K, 1), K)
  Z = [ Bool(rand(Bernoulli(v[k]))) for j in 1:J, k in 1:K ]
  mus = Dict([Bool(z) => sort(rand(c.mus_prior[z], L[z])) for z in 0:1])
  sig2 = [rand(c.sig2_prior) for i in 1:I]
  W = Matrix{Float64}(hcat([ rand(c.W_prior) for i in 1:I ]...)')
  lam = [ Int8.(rand(Categorical(W[i,:]), N[i])) for i in 1:I ]
  eta = begin
    function gen(z)
      arrMatTo3dArr([ rand(c.eta_prior[z]) for i in 1:I, j in 1:J ])
    end
    Dict([Bool(z) => gen(z) for z in 0:1])
  end
  gam = [zeros(Int8, N[i], J) for i in 1:I]
  for i in 1:I
    for j in 1:J
      for n in 1:N[i]
        z_lin = Z[j, lam[i][n]]
        gam[i][n, j] = rand(Categorical(eta[z_lin][i, j, :]))
      end
    end
  end

  return State(Z=Z, mus=mus, alpha=alpha, v=v, W=W, sig2=sig2, 
               eta=eta, lam=lam, gam=gam, y_imputed=y_imputed)
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
