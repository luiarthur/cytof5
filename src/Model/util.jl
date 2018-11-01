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

macro ifTrue(doThis, expr)
  return quote
    if $(esc(doThis))
      $(esc(expr))
    end
  end
end
