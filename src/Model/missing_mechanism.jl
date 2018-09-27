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
