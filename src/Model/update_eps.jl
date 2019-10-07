function update_eps!(i::Int, s::State, c::Constants, d::Data)
  a, b = params(c.eps_prior[i])
  num0 = sum(s.lam[i] .== 0)
  a += num0
  b += (d.N[i] - num0)
  s.eps[i] = rand(Beta(a, b))
end

function update_eps!(s::State, c::Constants, d::Data)
  for i in 1:d.I
    update_eps!(i, s, c, d)
  end
end
