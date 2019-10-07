function update_r!(s::StateFS, c::ConstantsFS, d::DataFS)
  for i in 1:d.data.I
    for k in 1:c.constants.K
      update_r(i, k, s, c, d)
    end
  end
end

function compute_w(ws::Vector{Float64}, r::Vector{Bool})
  wr = ws .* r
  return wr ./ sum(wr)
end

function logdmixture(z::Integer, i::Integer, n::Integer,
                     s::State, c::Constants, d::Data)::Float64
  return sum(logdmixture(z, i, n, j, s, c, d) for j in 1:d.J)
end

function update_r!(i::Integer, k::Integer,
                   s::StateFS, c::ConstantsFS, d::DataFS)

  function loglike(r_ik::Bool)::Float64
    ll = 0.0

    for n in 1:d.data.N[i]
      ri = s.r[i, :] .+ 0
      ri[k] = r_ik
      wr_i = compute_w(s.W_star[i, :], ri)
      non_zero_wri_idx = findall(wr_i .> 0)
      if length(non_zero_wri_idx) > 0
        ldmix = [begin 
                   z = s.theta.Z[j, k_prime]
                   logdmixture(z, i, n, s.theta, c.constants, d.data)
                 end for k_prime in non_zero_wri_idx]

        ll += MCMC.logsumexp(log.(wr_i[non_zero_wri_idx]) + ldmix)
      end
    end

    return ll
  end

  function logprior(r_ik::Bool)::Float64
    p_xi = compute_w(d.X[i, :], s.omega)
    return logpdf(Bernoulli(p_xi), r_ik)
  end

  log_prob(r_ik::Bool)::Float64 = logprior(r_ik) + loglike(r_ik)

  # Metropolis step
  cand = !s.r[i, k]  # Flip bit
  log_acceptance_ratio = log_prob(s.r[i, k]) - log_prob(cand)
  if log_acceptance_ratio > log(rand())
    s.r[i, k] = cand
  end
end
