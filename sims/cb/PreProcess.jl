module PreProcess
using Distributions

function isBadRow(x::Vector{T}; thresh::Float64=-6.0) where {T <: Number}
  return any(x .< thresh)
end

function removeBadRows!(y::Vector{Matrix{T}}; thresh::Float64=-6.0) where {T <: Number}
  I = length(y)
  N = size.(y, 1)

  goodRows = [[!isBadRow(y[i][n, :], thresh=thresh) for n in 1:N[i]] for i in 1:I]

  for i in 1:I
    y[i] = y[i][goodRows[i], :]
  end
end

#= Idea:
  - if across all samples, a marker has > 90% missing or negative, then remove
  - if across all samples, a marker has > 90% positive, then remove
=#

function isBadColumn(x::Vector{T};
                     maxNanOrNegProp::Float64=.9,
                     maxPosProp::Float64=.9)::Bool where {T <: Number}
  n = length(x)

  missOrNegProp = sum((isnan.(x)) .| (x .< 0)) / n
  posProp = sum(x .> 0) / n

  return missOrNegProp > maxNanOrNegProp || posProp > maxPosProp
end

function isBadColumnForAllSamples(j::Int, y::Vector{Matrix{T}};
                                  maxNanOrNegProp::Float64=.9,
                                  maxPosProp::Float64=.9) where {T}
  I = length(y)
  results = [isBadColumn(y[i][:, j], maxNanOrNegProp=maxNanOrNegProp, maxPosProp=maxPosProp) for i in 1:I]
  result = all(results)
  return result
end

function markerIsNearZero(j::Int, yi::Matrix{T}; thresh=0.5, prop=.25) where {T}
  out = (isnan.(yi[:, j]) .== false) .& (abs.(yi[:, j]) .< thresh)
  return mean(out) > prop
end

function markerIsNearZeroInAnySample(j::Int, y::Vector{Matrix{T}}; thresh=0.5, prop=.25) where T
  I = length(y)

  badMarker = [markerIsNearZero(j, y[i], thresh=thresh, prop=prop) for i in 1:I]
  return any(badMarker)
end

"""
subsample data if 0 < `subsample` < 1
returns goodColumns, new_J
"""
function preprocess!(y::Vector{Matrix{T}}; maxNanOrNegProp::Float64=.9,
                     maxPosProp::Float64=.9, subsample::Float64=0.0,
                     rowThresh::Float64=-Inf, 
                     noisyThresh=0.5, noisyProp=0.25) where {T}
  Js = size.(y, 2)
  J = Js[1]
  I = length(y)

  @assert all(J .== Js)
  @assert 0 <= subsample <= 1

  # Remove bad rows
  if !isinf(rowThresh)
    removeBadRows!(y, thresh=rowThresh) 
  end

  goodColumns = [!isBadColumnForAllSamples(j, y, maxNanOrNegProp=maxNanOrNegProp,
                                           maxPosProp=maxPosProp) for j in 1:J]

  goodColumns2=  [!markerIsNearZeroInAnySample(j, y, thresh=noisyThresh, prop=noisyProp)
                  for j in 1:J]

  goodColumns = goodColumns .& goodColumns2

  if 0 < subsample < 1
    for i in 1:I
      Ni = size(y[i], 1)
      new_Ni = Int(round(Ni * subsample))
      # random_indices = rand(1:Ni, new_Ni)
      random_indices = Distributions.sample(1:Ni, new_Ni, replace=false)
      y[i] = y[i][random_indices, :]
    end
  end

  for i in 1:I
    y[i] = y[i][:, goodColumns]
  end

  return goodColumns, J
end


end # PreProcess
