function update_sig2(z::Bool, i::Int, l::Int, s::State, c::Constants, d::Data)
  newShape = shape(c.sig2_prior)
  newScale = scale(c.sig2_prior)

  for j in 1:d.J
    for n in 1:d.N[i]
      k = s.lam[i][n]
      if k > 0
        if z == s.Z[j, k] && l == s.gam[i][n, j]
          newShape += 0.5
          newScale += (s.y_imputed[i][n, j] - mus(z, l, s, c, d)) ^ 2 / 2.0
        end
      end
    end
  end

  # newShape = shape(c.sig2_prior) + d.J * d.N[i] / 2
  # newScale = scale(c.sig2_prior) + ss / 2

  if c.sig2_range == [0, Inf]
    s.sig2[z][i, l] = rand(InverseGamma(newShape, newScale))
  else
    lower, upper = c.sig2_range
    s.sig2[z][i, l] = rand_uptrunc_ig(InverseGamma(newShape, newScale), upper)
  end
end

function update_sig2(s::State, c::Constants, d::Data)
  for z in 0:1
    for i in 1:d.I
      for l in 1:c.L[z]
        update_sig2(Bool(z), i, l, s, c, d)
      end
    end
  end
end
