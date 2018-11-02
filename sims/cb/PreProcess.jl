module PreProcess

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

"""
returns goodColumns, new_J
"""
function preprocess!(y::Vector{Matrix{T}}; maxNanOrNegProp::Float64=.9,
                     maxPosProp::Float64=.9) where {T}
  Js = size.(y, 2)
  J = Js[1]
  I = length(y)
  @assert all(J .== Js)

  goodColumns = [!isBadColumnForAllSamples(j, y, maxNanOrNegProp=maxNanOrNegProp,
                                           maxPosProp=maxPosProp) for j in 1:J]

  for i in 1:I
    y[i] = y[i][:, goodColumns]
  end

  return goodColumns, J
end


end # PreProcess