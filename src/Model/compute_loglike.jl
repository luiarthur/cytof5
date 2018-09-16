function compute_loglike(s::State, c::Constants, d::Data)
  ll = 0

  sumN = sum(d.N)

  for i in 1:d.I
    for j in 1:d.J
      for n in 1:d.N[i]
        z = s.Z[j, s.lam[i][n]]
        l = s.gam[i][n, j]
        ll += logpdf(Normal(s.mus[z][l], sqrt(s.sig2[i])), s.y_imputed[i][n, j]) / sumN
      end
    end
  end

  return ll
end
