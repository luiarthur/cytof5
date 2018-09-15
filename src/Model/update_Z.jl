function update_Z(s::State, c::Constants, d::Data)
  J = d.J
  K = c.K
  for j in 1:J
    for k in 1:K
      update_Zjk(s, c, d, j, k)
    end
  end
end


function update_Zjk(s::State, c::Constants, d::Data, j::Int, k::Int)
  ll0 = 0
  ll1 = 0

  for i in 1:d.I
    for n in 1:d.N[i]
      ll0 += dmixture(0, i, n, j, s, c, d)
      ll1 += dmixture(1, i, n, j, s, c, d)
    end
  end

  lp0 = log(s.v[k])
  lp1 = log(1 - s.v[k])
  p = 1 / (1 + exp((lp0 + ll0) - (lp1 + ll1)))

  s.Z[j, k] = p > rand() ? 1 : 0
end
