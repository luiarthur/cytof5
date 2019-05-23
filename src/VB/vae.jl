struct VAE{A <: AbstractArray}
  mean::A    # 1 x J
  log_sd::A  # 1 x J
end

# return standard deviation
_sd(vae::VAE) = exp.(vae.log_sd)
# _sd(vae::VAE) = .2 # This works. Maybe do this computation inline?

# return mean
_mean(vae::VAE) = vae.mean

# return (mean_function, sd_function)
function (vae::VAE)(yi_minibatch::Matrix, Ni::Integer)
  # Make a copy of yi_minibatch
  yi = deepcopy(yi_minibatch)
  m_mini = isnan.(yi)

  # set missing values to be 0 (to ensure not NaN)
  yi[m_mini] .= 0

  # mean function
  mean_fn = yi .* (1 .- m_mini) + _mean(vae) .* m_mini

  # sd function
  sd_fn = _sd(vae) .* m_mini

  # get random draw for imputed y (and observed y)
  z = randn(size(yi))
  yi_imputed = (z .* sd_fn) + mean_fn 

  @assert size(mean_fn) == size(sd_fn) == size(yi_imputed) == size(yi)

  # compute log_q(y_imputed | m_ini)
  log_qyi = sum(ADVI.lpdf_normal.(yi_imputed[m_mini], mean_fn[m_mini], sd_fn[m_mini]))
  log_qyi *= (Ni / size(yi, 1))

  # TODO: remove these
  @assert !isinf(log_qyi)
  @assert !isnan(log_qyi)
  
  return yi_imputed, log_qyi
end
