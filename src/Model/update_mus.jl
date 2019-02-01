function update_mus(s::State, c::Constants, d::Data, tuners::Tuners)
  # Update mus0
  # for l in c.L[0]:-1:1
  for l in 1:c.L[0]
    update_mus(0, l, s, c, d, tuners)
  end

  # Update mus1
  for l in 1:c.L[1]
    update_mus(1, l, s, c, d, tuners)
  end
end

function update_mus(z::Int, l::Int, s::State, c::Constants, d::Data, tuners::Tuners)
  (priorM, priorS, lower, upper) = params(priorMu(z, l, s, c))

  function lp(mus::Float64)::Float64
    out = logpdf(Normal(priorM, priorS), mus)

    if (z == 0 && l > 1)
      # NOTE: This should be log(cdf(N(pm, ps), mus) - cdf(N(pm, ps), lower))
      #       But this is good enough in practice.
      out -= logcdf(Normal(priorM, priorS), mus)
      # out -= MCMC.logsumexp(logcdf(Normal(priorM, priorS), mus),
      #                       logccdf(Normal(priorM, priorS), lower))
    elseif (z == 1 && l < c.L[1])
      # NOTE: This should be log(cdf(N(pm, ps), upper) - cdf(N(pm, ps), mus))
      #       But this is good enough in practice.
      out -= logccdf(Normal(priorM, priorS), mus)
      # out -= MCMC.logsumexp(logcdf(Normal(priorM, priorS), upper),
      #                       logccdf(Normal(priorM, priorS), mus))
    end

    return out
  end

  function ll(mus::Float64)::Float64
    out = 0.0
    for i in 1:d.I
      for n in 1:d.N[i]
        for j in 1:d.J
          k = s.lam[i][n]
          if s.gam[i][n, j] == l && k > 0 && s.Z[j, k] == z
            out += logpdf(Normal(mus, sqrt(s.sig2[i])), s.y_imputed[i][n, j])
          end
        end
      end
    end
    return out
  end

  s.mus[z][l] = MCMC.metLogitAdaptive(s.mus[z][l], ll, lp, tuners.mus[z][l],
                                      a=lower, b=upper)
  # println(s.mus)
end
