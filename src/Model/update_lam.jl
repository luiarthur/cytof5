# TODO: Check if this is correct! I'm not doing mixture here.
function update_lam(i::Int, n::Int, s::State, c::Constants, d::Data)
  priorVec = pdf.(Categorical(s.W[i,:]), 1:c.K)
  likeVec = ones(c.K)
  for k in 1:c.K
    for j in 1:d.J
      z = s.Z[j, k]
      l = s.gam[i][n, j]
      #likeVec[k] += pdf(Normal(s.mus[z][l], sqrt(s.sig2[i])), s.y_imputed[i][n, j])
      likeVec[k] *= dmixture(z, i, n, j, s, c, d)
    end
  end
  postVec = MCMC.normalize(priorVec .* likeVec)
  s.lam[i][n] = rand(Categorical(postVec))
end

function update_lam(s::State, c::Constants, d::Data)
  for i in 1:d.I
    for n in 1:d.N[i]
      update_lam(i, n, s, c, d)
    end
  end
end
