# TODO: Test
function prob_miss(y::Float64, b0::Float64, b1::Float64, b2::Float64)
  n = length(beta)
  return MCMC.sigmoid(b0 + b1 * y + b2 * y^2)
end

function solveB(yBounds::Vector{Float64}, pBounds::Vector{Float64})
  n = 3

  @assert length(yBounds) == n
  @assert length(pBounds) == n

  pLower, pCenter, pUpper = pBounds
  yLower, yCenter, yUpper = yBounds

  @assert pLower < pCenter > pUpper # Quadratic missing mechanism
  @assert yLower < yCenter < yUpper

  Y = [fill(1.0, n) (yBounds) (yBounds .^ 2)]
  beta = Y \ MCMC.logit.(pBounds)

  @assert all(abs.(MCMC.sigmoid.(Y * beta) - pBounds) .< 1E-10)

  return beta
end


