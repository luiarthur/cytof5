function update_v(k::Int, s::State, c::Constants, d::Data)
  sumZk = sum(s.Z[:, k])
  newA = s.alpha / c.K + sumZk
  newB = 1 + d.J - sumZk

  s.v[k] = rand(Beta(newA, newB))
end

function update_v(s::State, c::Constants, d::Data)
  for k in 1:c.K
    update_v(k, s, c, d)
  end
end
