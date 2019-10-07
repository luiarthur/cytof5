function update_eta!(i::Int, j::Int, s::State, c::Constants, d::Data)
  counts = Dict(z => zeros(c.L[z]) for z in 0:1)

  for n in 1:d.N[i]
    k = s.lam[i][n]
    if k > 0
      z = s.Z[j, k]
      l = s.gam[i][n, j]
      counts[z][l] += 1
    end
  end

  updatedParam0 = c.eta_prior[0].alpha .+ counts[0]
  updatedParam1 = c.eta_prior[1].alpha .+ counts[1]

  s.eta[0][i, j, :] = rand(Dirichlet(updatedParam0))
  s.eta[1][i, j, :] = rand(Dirichlet(updatedParam1))
end

function update_eta!(s::State, c::Constants, d::Data)
  for i in 1:d.I
    for j in 1:d.J
      update_eta!(i, j, s, c, d)
    end
  end
end
