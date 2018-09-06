module MCMC
using Distributions

export gibbs, TuningParam, metropolis, metropolisAdaptive, logpdfLogX1Param, logpdfLogX2Param, logpdfLogX3Param, logpdfLogX4Param

include("gibbs.jl")
include("TuningParam.jl")

function metropolis(curr::Float64, logFullCond::Function, stepSD::Float64)
  cand = curr + randn() * stepSD
  logU = log(rand())
  p = logFullCond(cand) - logFullCond(curr)
  out = p > logU ? cand : curr
  return out
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
function metropolisAdaptive(curr::Float64, logFullCond::Function, tuner::TuningParam;
                            delta::Function=n::Int64->min(n^(-0.3), 0.01), targetAcc::Float64=0.44)
  iter = tuner.currentIter
  factor = exp(delta(iter))

  if acceptanceRate(tuner) > targetAcc
    tuner.value *= factor
  else
    tuner.value /= factor
  end

  cand = rand(Normal(curr, tuner.value))
  logU = log(rand())
  p = logFullCond(cand) - logFullCond(curr)
  accept = p > logU

  update(tuner, accept)
  
  out = accept ? cand : curr

  return out
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

function logpdfLogX1Param(logX::Float64, logpdfX::Function, a::Float64)
  return logpdfX(exp(logX), a) + logX
end

function logpdfLogX2Param(logX::Float64, logpdfX::Function, a::Float64, b::Float64)
  return logpdfX(exp(logX), a, b) + logX
end

function logpdfLogX3Param(logX::Float64, logpdfX::Function, a::Float64, b::Float64, c::Float64)
  return logpdfX(exp(logX), a, b, c) + logX
end

function logpdfLogX4Param(logX::Float64, logpdfX::Function, a::Float64, b::Float64, c::Float64, d::Float64)
  return logpdfX(exp(logX), a, b, c, d) + logX
end

function logpdfLogInverseGamma(logX::Float64, a::Float64, b::Float64)
  return logpdfLogX2Param(logX, (x,aa,bb) -> logpdf(InverseGamma(aa, bb), x), a, b)
end

function logpdfLogitUniform(logitX::Float64, a::Float64, b::Float64)
  return logpdfLogX2Param(logitX, (x,aa,bb) -> logpdf(Uniform(aa, bb), x), a, b)
end

end # MCMC

