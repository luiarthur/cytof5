function update_eta(i::Int, j::Int, s::State, c::Constants, d::Data)
  # currParam = params(c.eta_prior)[1]
  currParam(z::Int) = params(c.eta_prior[z])[1]

  counts = Dict(z => zeros(c.L[z]) for z in 0:1)

  for n in 1:d.N[i]
    z = s.Z[j, s.lam[i][n]]
    l = s.gam[i][n, j]
    counts[z][l] += 1
  end

  updatedParam0 = currParam(0) .+ counts[0]
  updatedParam1 = currParam(1) .+ counts[1]

  s.eta[0][i, j, :] = rand(Dirichlet(updatedParam0))
  s.eta[1][i, j, :] = rand(Dirichlet(updatedParam1))
end

function update_eta(s::State, c::Constants, d::Data)
  for i in 1:d.I
    for j in 1:d.J
      update_eta(i, j, s, c, d)
    end
  end
end
