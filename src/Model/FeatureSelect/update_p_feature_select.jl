function compute_p(x::Vector{Float64}, omega::Vector{Float64})::Float64
  z = sum(x .* omega)
  return MCMC.sigmoid(z)
end
