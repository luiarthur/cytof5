# Update W
function update_W(s::StateFS, c::ConstantsFS, d::DataFS)
  for i in 1:d.data.I
    update_W(i, s, c, d)
  end
end


function update_W(i::Int, s::StateFS, c::ConstantsFS, d::DataFS)
  wr_i = s.W_star[i, :] .* s.r[i, :]
  s.theta.W[i, :] .= wr_i ./ sum(wr_i, dims=2)
end


# Update W star
function update_W_star(s::StateFS, c::ConstantsFS, d::DataFS, tuners::TunersFS)
  for i in 1:d.data.I
    for k in 1:c.constants.K
      update_W_star(i, k, s, c, d, tuners)
    end
  end
end


function update_W_star(i::Int, k::Int, s::StateFS, c::ConstantsFS, d::DataFS,
                       tuners::TunersFS)
  if s.r[i, k] == 0
    # Sample from prior
    s.W_star[i, k] = rand(c.W_star_prior)
  else  # s.r[i, k] == 1
    logprior(w::Float64) = logpdf(c.W_star_prior, w)

    function loglike(w::Float64)::Float64
      log_numer = sum(s.theta.lam[i] .== k) * log(w)
      w_other = s.W_star[i, 1:end .!= k]
      r_other = s.r[i, 1:end .!= k]
      log_denom = c.constants.N[i] * log(w + sum(w_other .* r_other))
      return log_numer - log_denom
    end

    # Metropolis step to update W*_{ik}
    s.W_star[i, k] = MCMC.metLogAdaptive(s.W_star[i, k],
                                         loglike, logprior, tuners)
  end
end
