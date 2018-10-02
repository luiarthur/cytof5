module MCMC
using Distributions

export gibbs, TuningParam, metropolis, metropolisAdaptive, logpdfLogX, metLogAdaptive, metLogitAdaptive

include("gibbs.jl")
include("TuningParam.jl")
include("Conjugate.jl")

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
                            delta::Function=n::Int->min(n^(-0.5), 0.01),
                            targetAcc::Float64=0.44)
  iter = tuner.currentIter
  batch_size = tuner.batch_size

  if iter % batch_size == 0
    n = Int(floor(iter / batch_size))
    factor = exp(delta(n))
    if acceptanceRate(tuner) > targetAcc
      tuner.value *= factor
    else
      tuner.value /= factor
    end

    tuner.acceptanceCount = 0
  end

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
                          tuner::TuningParam; a::Float64=0, b::Float64=1, 
                          delta::Function=n::Int64->min(n^(-0.5), 0.01),
                          targetAcc::Float64=0.44)

  function lfc_logitX(logit_x::Float64)
    x = sigmoid(logit_x, a=a, b=b)
    #lp_logitX = lp(x) + logpdf(Logistic(), logit_x)
    lp_logitX = logpdfLogitX(logit_x, lp, a, b)
    return ll(x) + lp_logitX
  end

  logit_x = metropolisAdaptive(logit(curr,a=a,b=b), lfc_logitX, tuner,
                               delta=delta, targetAcc=targetAcc)

  return sigmoid(logit_x, a=a, b=b)
end

function metLogAdaptive(curr::Float64, ll::Function, lp::Function,
                        tuner::TuningParam;
                        delta::Function=n::Int64->min(n^(-0.5), 0.01),
                        targetAcc::Float64=0.44)

  function lfc_logX(log_x::Float64)
    x = exp(log_x)
    return ll(x) + logpdfLogX(log_x, lp)
  end

  log_x = metropolisAdaptive(log(curr), lfc_logX, tuner,
                             delta=delta, targetAcc=targetAcc)

  return exp(log_x)
end

function normalize(x::Vector{T}) where T
  return isprobvec(x) ? x : x / sum(x)
end

end # MCMC

