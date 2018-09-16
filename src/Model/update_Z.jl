function update_Zjk(s::State, c::Constants, d::Data, j::Int, k::Int)
  ll0 = 0
  ll1 = 0

  for i in 1:d.I
    for n in 1:d.N[i]
      if s.lam[i][n] == k
        ll0 += log(dmixture(0, i, n, j, s, c, d))
        ll1 += log(dmixture(1, i, n, j, s, c, d))
      end
    end
  end

  lp0 = log(1 - s.v[k])
  lp1 = log(s.v[k])
  p = 1 / (1 + exp((lp0 + ll0) - (lp1 + ll1)))

  s.Z[j, k] = rand(Bernoulli(p))
end

function update_Z(s::State, c::Constants, d::Data)
  for j in 1:d.J
    for k in 1:c.K
      update_Zjk(s, c, d, j, k)
    end
  end
end

