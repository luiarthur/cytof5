function update_W(i::Int, s::State, c::Constants, d::Data)
  currParam = params(c.W_prior)[1]
  counts = zeros(c.K)
  for n in 1:d.N[i]
    k = s.lam[i][n]
    counts[k] += 1
  end
  updatedParam = currParam .+ counts
  s.W[i, :] = rand(Dirichlet(updatedParam))
end

function update_W(s::State, c::Constants, d::Data)
  for i in 1:d.I
    update_W(i, s, c, d)
  end
end
