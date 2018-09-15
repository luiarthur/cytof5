function update_sig2(i::Int, s::State, c::Constants, d::Data)
  ss = 0
  for j in 1:d.J
    for n in 1:d.N[i]
      z = s.Z[j, s.lam[i][n]]
      l = s.gam[i][n, j]
      ss += (s.y_imputed[i][n, j] - s.mus[z][l]) ^ 2
    end
  end

  newShape = shape(c.sig2_prior) + d.J * d.N[i] / 2
  newScale = scale(c.sig2_prior) + ss / 2
  s.sig2[i] = rand(InverseGamma(newShape, newScale))
end

function update_sig2(s::State, c::Constants, d::Data)
  for i in 1:d.I
    update_sig2(i, s, c, d)
  end
end
