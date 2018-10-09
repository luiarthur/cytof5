### DIC ###

# TODO: Test
function deviance(param::T, loglike::Function) where T
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
- updateParams(d::DICstream{T}, param::P) where {T, P}, to add param to d.paramSum
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
    updateParams{T, P}(x, param)
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

