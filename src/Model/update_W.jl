function update_W(i::Int, s::State, c::Constants, d::Data)
  currParam = c.W_prior.alpha
  counts = zeros(c.K)
  for n in 1:d.N[i]
    k = s.lam[i][n]
    if k > 0
      counts[k] += 1
    end
  end
  updatedParam = currParam .+ counts
  s.W[i, :] = rand(Dirichlet(updatedParam))
end

function update_W(s::State, c::Constants, d::Data)
  for i in 1:d.I
    update_W(i, s, c, d)
  end
end
