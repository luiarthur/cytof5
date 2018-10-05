# Functions for measuring model fit

### DIC ###

# TODO: Test
function deviance(param::Any, loglike::Function)
  return -2 * loglike(param)
end

# TODO: Test
function DIC(params::Vector{T}, loglike::Function, paramMean::Function) where T
  D(param) = deviance(param, loglike)
  
  Dbar = mean(D.(params))
  paramBar = paramMean(params)
  pD = Dbar - D(paramBar)

  return Dbar + pD
end

# TODO: Test
"""
Methods to implement:
- updateParams(d::DICstream{T}, param::T) where T, to add param to d.paramSum
- paramMeanCompute(d::DICstream{T}) where T, to compute the mean of the parameters
"""
mutable struct DICstream{T}
  paramSum::T
  Dsum::Float64
  counter::Int
  loglike::Function

  DICstream{T}(tmp, loglike) where {T} = new(tmp, 0, 0, loglike)
end

# TODO: Test
function updateDIC(x::DICstream{T}, param::T) where {T}
  # update Dsum
  x.Dsum += deviance(param, x.loglike)

  # update paramSum
  if x.counter == 0
    x.paramSum = deepcopy(param)
  else
    updateParams{T}(x, param)
  end

  # update counter
  x.counter += 1
end

# TODO: Test
"""
DIC is a measure of fit. Higher is better.

DIC = Dmean + pD, where
- Dmean is a measure of fit. Higher is better.
- pD is a measure of (non) complexity. Higher is better.
"""
function computeDIC(x::DICstream{T}, return_Dmean_pD::Bool=false) where {T}
  Dmean = x.Dsum / x.counter
  paramMean = paramMeanCompute(x)

  if return_Dmean_pD
    pD = Dmean - deviance(paramMean, x.loglike)
    return Dmean, pD
  else
    return 2 * Dmean - deviance(paramMean, x.loglike)
  end
end

### CPO ###
harmonicMean(x) = 1 / mean(1 / x)

"""
CPO for one data point

likelihood: likelihood for that data point. Note that this should take one sample of the parameters and return the likelihood, and NOT the log-likelihood.

http://webpages.math.luc.edu/~ebalderama/myfiles/modelchecking101_pres.pdf
"""

function cpo(params::Vector{T}, likelihood::Function) where T
  return harmonicMean(likelihood.(params))
end

"""
Methods to implement:
- updateInvLikelihood(c::CPOstream{T}, invLike::T) where T, to add invLikelihoods to c.invLikelihoodSum
"""
mutable struct CPOstream{T}
  invLikelihoodSum::T
  counter::Int

  CPOstream{T}(tmp) where T = new(tmp, 0)
end

function updateInvLikelihood(c::CPOstream{Vector{Array{T,N}}}, invLike::Vector{Array{T,N}}) where {T, N}
  c.invLikelihoodSum += invLike
end

function updateInvLikelihood(c::CPOstream{Array{T, N}}, invLike::Array{T, N}) where {T, N}
  c.invLikelihoodSum += invLike
end

function updateCPO(x::CPOstream, invLike::T) where T
  # Update invLikelihoodSum
  if x.counter == 0
    x.invLikelihoodSum = deepcopy(invLike)
  else
    updateInvLikelihood(x, invLike)
  end

  # Update counter
  x.counter += 1
end

function computeCPO(x::CPOstream{Array{T, N}}) where {T, N}
  return x.counter ./ x.invLikelihoodSum
end

function computeCPO(x::CPOstream{Vector{Array{T, N}}}) where {T, N}
  return [ x.counter ./ invLikeSum for invLikeSum in x.invLikelihoodSum ]
end

### LPML ###
function computeLPML(x::CPOstream{Array{T, N}}; verbose::Int=1) where {T, N}
  return mean(log.(computeCPO(x)))
end

function computeLPML(x::CPOstream{Vector{Array{T, N}}}; verbose::Int=1) where {T, N}
  cpos = computeCPO(x)
  logCpos = [ log.(cpo) for cpo in cpos ]
  logCpos = vcat(vec.(logCpos)...)
  logCposSafe = filter(lcpo -> lcpo > -Inf, logCpos)

  numNegInf = length(logCpos) - length(logCposSafe)
  if numNegInf > 0 && verbose > 0
    println(" -- Warning: there were $numNegInf -Inf in log CPO's")
    if verbose > 1
      idx = [ findall(c -> log.(c) .== -Inf, cpo) for cpo in cpos ]
      for i in 1:length(idx)
        println("$i, $(cpos[i][idx[i]])")
        println("$i, $(idx[i])")
      end
    end
    x.counter = 0
  end

  return mean(logCposSafe)
end


