function print_debug_Z(i::Int, n::Int, j::Int, s::State, c::Constants, d::Data)
  println("y_inj = $(s.y_imputed[i][n, j])")
  println("sig2[i]: $(s.sig2[i])")
  println("mus0: $(mus(false, s, c, d))")
  println("mus1: $(mus(true, s, c, d))")
  println("etaz0_i$(i)_j$(j): $(s.eta[0][i, j, :])")
  println("etaz1_i$(i)_j$(j): $(s.eta[1][i, j, :])")
end

function update_Z(s::State, c::Constants, d::Data)
  ll0 = zeros(d.J, c.K)
  ll1 = zeros(d.J, c.K)

  for i in 1:d.I
    for n in 1:d.N[i]
      k = s.lam[i][n]
      if k > 0
        for j in 1:d.J
          # ll0[j, k] += log(dmixture(0, i, n, j, s, c, d))
          # ll1[j, k] += log(dmixture(1, i, n, j, s, c, d))
          ll0[j, k] += logdmixture(false, i, n, j, s, c, d)
          ll1[j, k] += logdmixture(true, i, n, j, s, c, d)
        end
      end
    end
  end

  for k in 1:c.K
    # lp0 = log1p(-s.v[k])
    # lp1 = log(s.v[k])
    b = cumprod(s.v)
    lp0 = log1p(-b[k])
    lp1 = log(b[k])

    for j in 1:d.J
      lfc0 = lp0 + ll0[j, k]
      lfc1 = lp1 + ll1[j, k]
      p = 1.0 / (1.0 + exp(lfc0 - lfc1))
      if isnan(p)
        println("WARNING in update_Z: p = NaN.")
      end
      s.Z[j, k] = p > rand() # rand(Bernoulli(p))
    end
  end
end
