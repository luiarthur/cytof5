# TODO: Test
function prob_miss(y::AbstractFloat, beta::Vector{Float64})
  n = length(beta)
  x = sum((y .^ (0:n-1)) .* beta)
  return MCMC.sigmoid(x)
end

function assertValidBounds(y::Vector{Float64}, p::Vector{Float64})
  n = length(y)
  @assert length(y) == length(p) == n

  @assert 2 <= n <= 3

  if n == 3
    @assert p[1] < p[2] > p[3] # Quadratic missing mechanism
    @assert y[1] < y[2] < y[3]
  elseif n ==2
    @assert p[1] > p[2] # Linear missing mechanism
    @assert y[1] < y[2]
  else
    println("In assertValidBounds: This should never be printed!")
    @assert false
  end
end

function solveBeta(y::Vector{Float64}, p::Vector{Float64})
  assertValidBounds(y, p)

  n = length(y)
  Y = [fill(1.0, n) (y) (y .^ 2)]
  beta = Y \ MCMC.logit.(p)

  @assert all(abs.(MCMC.sigmoid.(Y * beta) - p) .< 1E-10)

  return beta
end
