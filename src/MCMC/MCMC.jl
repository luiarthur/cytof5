module MCMC
using Distributions

export gibbs, TuningParam, metropolis, metropolisAdaptive, logpdfLogX, metLogAdaptive, metLogitAdaptive

include("gibbs.jl")
include("TuningParam.jl")
include("Conjugate.jl")
include("Fitness.jl")
include("Util.jl")

function metropolisBase(curr::Float64, logFullCond::Function, stepSD::Float64)
  cand = curr + randn() * stepSD
  logU = log(rand())
  logP = logFullCond(cand) - logFullCond(curr)
  accept = logP > logU
  draw = accept ? cand : curr
  return (draw, accept)
end

function metropolis(curr::Float64, logFullCond::Function, stepSD::Float64)
  return metropolisBase(curr, logFullCond, stepSD)[1]
end

function metropolis(curr::Vector{Float64}, logFullCond::Function, stepSD::Matrix{Float64})
  cand = rand(MvNormal(curr, stepSD))
  logU = log(rand())
  p = logFullCond(cand) - logFullCond(curr)
  out = p > logU ? cand : curr
  return out
end

"""
Adaptive metropolis (within Gibbs). See section 3 of the paper below:
  http://probability.ca/jeff/ftpdir/adaptex.pdf

Another useful website:
  https://m-clark.github.io/docs/ld_mcmc/index_onepage.html
"""
function metropolisAdaptive(curr::Float64, logFullCond::Function,
                            tuner::TuningParam;
                            update::Function=update_tuning_param_default)
  draw, accept = metropolisBase(curr, logFullCond, tuner.value)
  update(tuner, accept)
  return draw
end

function logit(p::Float64; a::Float64=0.0, b::Float64=1.0)
  return log(p - a) - log(b - p)
end

function sigmoid(x::Float64; a::Float64=0.0, b::Float64=1.0)
  out = 0.0

  if a == 0 && b == 1 
    out = 1 / (1 + exp(-x))
  else
    ex = exp(x)
    out = (b * ex + a) / (1 + ex)
  end

  return out
end

function logpdfLogX(logX::Float64, logpdfX::Function)
  return logpdfX(exp(logX)) + logX
end

function logpdfLogitX(logitX::Float64, logpdfX::Function, a::Float64, b::Float64)
  x = sigmoid(logitX, a=a, b=b)
  logJacobian = logpdf(Logistic(), logitX) + log(b - a)
  logpdfX(x) + logJacobian
end

function metLogitAdaptive(curr::Float64, ll::Function, lp::Function,
                          tuner::TuningParam; a::Float64=0.0, b::Float64=1.0, 
                          update::Function=update_tuning_param_default)

  function lfc_logitX(logit_x::Float64)
    x = sigmoid(logit_x, a=a, b=b)
    #lp_logitX = lp(x) + logpdf(Logistic(), logit_x)
    lp_logitX = logpdfLogitX(logit_x, lp, a, b)
    return ll(x) + lp_logitX
  end

  logit_x = metropolisAdaptive(logit(curr,a=a,b=b), lfc_logitX, tuner, update=update)

  return sigmoid(logit_x, a=a, b=b)
end

function metLogAdaptive(curr::Float64, ll::Function, lp::Function,
                        tuner::TuningParam;
                        update::Function=update_tuning_param_default)

  function lfc_logX(log_x::Float64)
    x = exp(log_x)
    return ll(x) + logpdfLogX(log_x, lp)
  end

  log_x = metropolisAdaptive(log(curr), lfc_logX, tuner, update=update)

  return exp(log_x)
end

function normalize(x::Vector{T})::Vector{T} where T
  if isprobvec(x)
    return x
  elseif length(x) == 1
    @assert x[1] > 0
    return [1.0]
  else
    out = x / sum(x)
    out[1] = 1.0 - sum(x[2:end])
    return out
  end
end


"""
weighted sampling: takes (unnormalized) log probs and returns index
"""
function wsample_logprob(logProbs::Vector{T}) where {T <: Number}
  log_p_max = maximum(logProbs)
  p = exp.(logProbs .- log_p_max)
  return Distributions.wsample(p)
end

function logsumexp(logx::Vector{T}) where {T <: Number}
  mx = maximum(logx)
  return log(sum(exp.(logx .- mx))) + mx
end

function logsumexp(logx::T...) where {T <: Number}
  mx = maximum(logx)
  return log(sum(exp.(logx .- mx))) + mx
end


end # MCMC

