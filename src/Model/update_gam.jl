function update_gam(i::Int, n::Int, j::Int, s::State, c::Constants, d::Data)
  z = s.Z[j, s.lam[i][n]]
  priorVec = pdf(Categorical(s.eta[z][i, j, :]), s.gam[i][n, j])  
  likeVec = pdf.(Normal.(s.mus[z], sqrt(s.sig2[i])), s.y_imputed[i][n, j])
  postVec = MCMC.normalize(priorVec .* likeVec)
  s.gam[i][n, j] = rand(Categorical(postVec))
end

function update_gam(s::State, c::Constants, d::Data)
  for i in 1:d.I
    for j in 1:d.J
      for n in 1:d.N[i]
        update_gam(i, n, j, s, c, d)
      end
    end
  end
end

