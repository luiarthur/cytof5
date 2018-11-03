function padZeroCols(x::Matrix, desiredSize::Int)
  @assert size(x, 2) <= desiredSize

  n = size(x, 1)

  if size(x, 2) == desiredSize
    return x
  else
    return padZeroCols([x zeros(n)], desiredSize)
  end
end

#= Test
padZeroCols(randn(3,5), 10)
=#

macro doIf(condition, expr)
  return quote
    if $(esc(condition))
      $(esc(expr))
    end
  end
end

macro leftTrunc!(minVal, x)
  return quote
    if $(esc(x)) < $(esc(minVal))
      $(esc(x)) = $(esc(minVal))
    end
  end
end
