function update_y_imputed(i::Int, n::Int, j::Int, s::State, c::Constants, d::Data, tuners::Tuners)

  function logFullCond(yinj)
    p = prob_miss(yinj, s.b0[i], s.b1[i])
    z = s.Z[j, s.lam[i][n]]
    l = s.gam[i][n, j]
    mu = s.mus[z][l]
    sig = sqrt(s.sig2[i])
    logPrior = logpdf(Normal(mu, sig), yinj)
    return log(p) + logPrior
  end
  
  s.y_imputed[i][n, j] = MCMC.metropolisAdaptive(s.y_imputed[i][n, j],
                                                 logFullCond,
                                                 tuners.y_imputed[i, n, j])
end

function update_y_imputed(s::State, c::Constants, d::Data, tuners::Tuners)
  for i in 1:d.I
    for j in 1:d.J
      for n in 1:d.N[i]
        if d.m[i][n, j] == 1
          update_y_imputed(i, n, j, s, c, d, tuners)
        end
      end
    end
  end
end
