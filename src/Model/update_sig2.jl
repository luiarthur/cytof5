function update_sig2(i::Int, s::State, c::Constants, d::Data)
  ss = 0.0

  for j in 1:d.J
    for n in 1:d.N[i]
      k = s.lam[i][n]
      if k > 0
        z = s.Z[j, k]
        l = s.gam[i][n, j]
        ss += (s.y_imputed[i][n, j] - s.mus[z][l]) ^ 2
      end
    end
  end

  newShape = shape(c.sig2_prior) + d.J * d.N[i] / 2
  newScale = scale(c.sig2_prior) + ss / 2

  if c.sig2_range == [0, Inf]
    s.sig2[i] = rand(InverseGamma(newShape, newScale))
  else
    lower, upper = c.sig2_range
    s.sig2[i] = rand_uptrunc_ig(InverseGamma(newShape, newScale), upper)
  end
end

function update_sig2(s::State, c::Constants, d::Data)
  for i in 1:d.I
    update_sig2(i, s, c, d)
  end
end
