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

