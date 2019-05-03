# TODO: Test

mutable struct VAE{A <: AbstractArray}
  mean::A    # I x J
  log_sd::A  # I x J
end

# return standard deviation
_sd(vae::VAE, i::Integer) = exp.(vae.log_sd[i:i, :])

# return mean
_mean(vae::VAE, i::Integer) = vae.mean[i:i, :]

# return (mean_function, sd_function)
function (vae::VAE)(i::Integer, yi_minibatch::Matrix)
  # Make a copy of yi_minibatch
  yi = deepcopy(yi_minibatch)
  m_mini = isnan.(yi)

  # set missing values to be 0 (to ensure not NaN)
  yi[m_mini] .= 0

  # mean function
  mean_fn = yi .* (1 .- m_mini) .+ _mean(vae, i) .* m_mini

  # sd function
  sd_fn = _sd(vae, i) .* m_mini

  # get random draw for imputed y (and observed y)
  z = randn(size(yi))
  yi_imputed = z .* sd_fn .+ mean_fn 

  # compute log_q(y_imputed | m_ini)
  log_qyi = sum(ADVI.lpdf_normal.(yi_imputed[m_mini], mean_fn[m_mini], sd_fn[m_mini]))
  
  return yi_imputed, log_qyi
end
