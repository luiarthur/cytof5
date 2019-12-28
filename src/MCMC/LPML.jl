### CPO ###
harmonicMean(x) = 1 / mean(1 / x)

"""
CPO for one data point

likelihood: likelihood for that data point. Note that this should take one
sample of the parameters and return the likelihood, and NOT the log-likelihood.

http://webpages.math.luc.edu/~ebalderama/myfiles/modelchecking101_pres.pdf
"""

function cpo(params::Vector{T}, likelihood::Function) where T
  return harmonicMean(likelihood.(params))
end

mutable struct CPOstream{T <: AbstractFloat}
  invLikelihoodSum::Vector{T}
  counter::Int

  CPOstream{T}() where T = new(T[], 0)
end

function updateCPO(cs::CPOstream{T}, like::Vector{T}) where {T <: AbstractFloat}
  @assert cs.counter >= 0

  # Either initialize or Update inverse-likelihood sum
  if cs.counter == 0
    # Initialize CPO stream
    cs.invLikelihoodSum = 1 ./ like
  else
    # Update invLikelihoodSum
    cs.invLikelihoodSum += 1 ./ like
  end

  # Update counter
  cs.counter += 1
end

"""
Computes (elementwise) log CPO for each observation.
"""
function computeLogCPO(cs::CPOstream{T}) where {T <: AbstractFloat}
  # NOTE: Equivalent to `log(cs.counter ./ cs.invLikelihoodSum)`
  return log(cs.counter) .- log.(cs.invLikelihoodSum)
end

### LPML ###
function computeLPML(cs::CPOstream{T}; verbose::Int=1) where {T <: AbstractFloat}
  mean_log_cpo = mean(computeLogCPO(cs))

  if isinf(mean_log_cpo) || isnan(mean_log_cpo)
    # Reset LPML
    cs.counter = 0
  end

  return mean_log_cpo
end
