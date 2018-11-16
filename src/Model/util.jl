import Base.show

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

# Pretty printing of InverseGamma
function show(io::IO, x::Distributions.InverseGamma)
  print(io, "InverseGamma(shape=$(shape(x)), scale=$(scale(x)))")
end

# Pretty printing of Gamma
function show(io::IO, x::Distributions.Gamma)
  print(io, "Gamma(shape=$(shape(x)), rate=$(rate(x)))")
end


"""
log info with println and flush
"""
function logger(x; newline=true)
  if newline
    println(x)
  else
    print(x)
  end
  flush(stdout)
end

