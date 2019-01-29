function update_mus(s::State, c::Constants, d::Data, tuners::Tuners)
  for z in 0:1
    for l in 1:c.L[z]
      update_mus(z, l, s, c, d, tuners)
    end
  end
end

function update_mus(z::Int, l::Int, s::State, c::Constants, d::Data, tuners::Tuners)
  (priorM, priorS, lower, upper) = params(priorMu(z, l, s, c))

  function lp(mus::Float64)::Float64
    out = -(mus - priorM) ^ 2 / (2 * priorS ^ 2)

    if z == 0 && l > 1
      m0_min = c.mus_prior[0].lower
      out -= MCMC.logsumexp(logpdf(Normal(priorM, priorS), mus),
                            -logpdf(Normal(priorM, priorS), m0_min))
    elseif z == 1 && l < c.L[1]
      m1_max = c.mus_prior[1].upper
      out -= MCMC.logsumexp(logpdf(Normal(priorM, priorS), m1_max),
                            -logpdf(Normal(priorM, priorS), mus))
    end

    return out
  end

  function ll(mus::Float64)::Float64
    out = 0.0
    for i in 1:d.I
      for n in 1:d.N[i]
        for j in 1:d.J
          if s.gam[i][n, j] == l && s.lam[i][n] > 0 
            out += logpdf(Normal(mus, sqrt(s.sig2[i])), s.y_imputed[i][n, j])
          end
        end
      end
    end
    return out
  end

  s.mus[z][l] = MCMC.metLogitAdaptive(s.mus[z][l], ll, lp, tuners.mus[z][l],
                                      a=lower, b=upper)
end
