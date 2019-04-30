# TODO: Test

mutable struct VAE{A <: AbstractArray}
  mean::A    # I x J
  log_sd::A  # I x J
end

# return standard deviation
sd(vae::VAE, i::Integer) = exp.(vae.log_sd[i:i, :])

# return mean
mean(vae::VAE, i::Integer) = vae.mean[i:i, :]

# return (mean_function, sd_function)
function (vae::VAE)(i::Integer, y_mini::Matrix, m_mini::Matrix)
  # Make a copy of y_mini
  y = deepcopy(y_mini)

  # set missing values to be 0 (to ensure not NaN)
  y[m_mini] .= 0

  # mean function
  mean_fn = y .* (1 .- m_mini) .+ mean(vae, i) .* m_mini

  # sd function
  sd_fn = sd(vae, i) .* m_mini

  # get random draw for imputed y (and observed y)
  z = randn(size(y))
  y_imputed = z .* sd_fn .+ mean_fn 

  # compute log_q(y_imputed | m_ini)
  log_q = sum(lpdf_normal.(y_imputed[m_mini], mean_fn[m_mini], sd_fn[m_mini]))
  
  return y_imputed, log_q
end
