### DIC ###

# Batch DIC
function deviance(param::T, loglike::Function) where T
  return -2 * loglike(param)
end

function DIC(params::Vector{T}, loglike::Function, paramMean::Function) where T
  D(param) = deviance(param, loglike)
  
  Dbar = mean(D.(params))
  paramBar = paramMean(params)
  pD = Dbar - D(paramBar)

  return Dbar + pD
end

# TODO: Test
# Streaming DIC
"""
Methods to implement:
- updateParams(d::DICstream{T}, param::P) where {T, P}, to add param to d.paramSum
- paramMeanCompute(d::DICstream{T}) where T, to compute the mean of the parameters
- loglike(param::P) computes the loglikelihood based on the parameters (param)
- convert(state::State) convets the current state to param::Param
"""
mutable struct DICstream{T}
  paramSum::T
  Dsum::Float64
  counter::Int

  DICstream{T}(tmp) where T = new(tmp, 0, 0)
end

# TODO: Test
"""
convert: converts state to param
"""
function updateDIC(x::DICstream{T}, state::S, updateParams::Function, loglike::Function, convert::Function) where {T, S}
  param = convert(state)

  # update Dsum
  x.Dsum += deviance(param, loglike)

  # update paramSum
  if x.counter > 0
    updateParams(x, param)
  else
    @assert x.counter == 0
    x.paramSum = deepcopy(param)
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
function computeDIC(x::DICstream{T}, loglike::Function, paramMeanCompute::Function;
                    return_Dmean_pD::Bool=false) where {T}
  Dmean = x.Dsum / x.counter
  paramMean = paramMeanCompute(x)
  pD = Dmean - deviance(paramMean, loglike)

  if return_Dmean_pD
    return Dmean, pD
  else
    return Dmean + pD
  end
end

