function compute_p(x::Vector{Float64}, omega::Vector{Float64})::Float64
  z = sum(x .* omega)
  return MCMC.sigmoid(z)
end


function compute_p(x::Matrix{Float64}, omega::Vector{Float64})
  z = x * omega
  return MCMC.sigmoid.(z)
end

