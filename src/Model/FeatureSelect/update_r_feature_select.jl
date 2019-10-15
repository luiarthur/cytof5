function update_r!(s::StateFS, c::ConstantsFS, d::DataFS)
  # update each r_{i, k}
  for i in 1:d.data.I
    for k in 1:c.constants.K
      update_r!(i, k, s, c, d)
    end
  end
end


function compute_w(ws::Vector{Float64}, r::Vector{Bool})
  wr = ws .* r
  return wr ./ sum(wr)
end


function logdmixture_r(i::Integer, n::Integer, k::Integer,
                       s::State, c::Constants, d::Data)::Float64
  return sum(logdmixture(s.Z[j, k], i, n, j, s, c, d) for j in 1:d.J)
end


function update_r!(i::Integer, k::Integer,
                   s::StateFS, c::ConstantsFS, d::DataFS)

  # Log likelihood as a function of `r_{i, k}`
  function loglike(r_ik::Bool)::Float64
    # Initialize log likelihood
    ll = 0.0

    for n in 1:d.data.N[i]
      # Make a copy of the current (k-dim) vector r_i
      ri = s.r[i, :]

      # Replace r_{i, k} with the provided `r_ik`
      ri[k] = r_ik

      # Compute W given W* and the updated r_i
      wi = compute_w(s.W_star[i, :], ri)

      # Get the indices of the W_i such that W_{i, k} > 0
      non_zero_wri_idx = findall(wi .> 0)

      if length(non_zero_wri_idx) > 0  # i.e. if any W_i > 0
        ldmix = [logdmixture_r(i, n, k_prime, s.theta, c.constants, d.data)
                 for k_prime in non_zero_wri_idx]

        # Add to log likelihood
        ll += MCMC.logsumexp(log.(wi[non_zero_wri_idx]) + ldmix)
      end
    end

    return ll
  end

  # Log prior as a function of `r_{i, k}`
  function logprior(r_ik::Bool)::Float64
    p_xi = compute_p(d.X[i, :], s.omega)
    return logpdf(Bernoulli(p_xi), r_ik)
  end

  # Log full conditional as a function of `r_{i, k}`
  log_prob(r_ik::Bool)::Float64 = logprior(r_ik) + loglike(r_ik)

  # update r_{i, k} with a metropolis step
  rik_metropolis_update!(i, k, log_prob, s)
end


function rik_metropolis_update!(i::Integer, k::Integer,
                                log_prob::Function, s::StateFS)
  cand = !s.r[i, k]  # Flip bit
  log_acceptance_ratio = log_prob(cand) - log_prob(s.r[i, k])

  # accept with probability `accecptance_ratio`
  if log_acceptance_ratio > log(rand())
    s.r[i, k] = cand
  end
end
