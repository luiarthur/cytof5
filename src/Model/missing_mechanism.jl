function prob_miss(y, b0, b1)
  return MCMC.sigmoid(b0 - b1 * y)
end

function solveB(yBounds::Tuple, pBounds::Tuple)
  pLower, pUpper = pBounds
  yLower, yUpper = yBounds
  @assert pLower > pUpper # probability of missing at smaller y values should be higher
  @assert yLower < yUpper

  b1 = (MCMC.logit(pLower) - MCMC.logit(pUpper)) / (yUpper - yLower)
  b0 = MCMC.logit(pLower) + b1 * yLower

  return (b0, b1)
end

# function prob_miss(y::T, beta::Vector{B}::T) where {T <: Number, B <: Number}
#   n = length(beta)
#   ys = y .^ collect(0:(n-1))
#   return MCMC.sigmoid(sum(beta .* ys))
# end
# 
# function solveB(yBounds::Tuple, pBounds::Tuple)
#   pLower, pCenter, pUpper = pBounds
#   yLower, yCenter, yUpper = yBounds
#   @assert pLower < pCenter > pUpper # Quadratic missing mechanism
#   @assert yLower < yCenter < yUpper
# 
#   b1 = (MCMC.logit(pLower) - MCMC.logit(pUpper)) / (yUpper - yLower)
#   b0 = MCMC.logit(pLower) + b1 * yLower
# 
#   return (b0, b1)
# end

