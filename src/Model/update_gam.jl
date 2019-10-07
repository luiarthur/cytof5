function update_gam!(i::Int, n::Int, j::Int, s::State, c::Constants, d::Data)
  k = s.lam[i][n]
  if k > 0
    z = s.Z[j, k]
    logpriorVec = log.(s.eta[z][i, j, :])
    loglikeVec = logpdf.(Normal.(mus(z, s, c, d), sqrt(s.sig2[i])), s.y_imputed[i][n, j])
    logPostVec = logpriorVec .+ loglikeVec
    s.gam[i][n, j] = MCMC.wsample_logprob(logPostVec)
  else
    s.gam[i][n, j] = 0
  end
end

function update_gam!(s::State, c::Constants, d::Data)
  for i in 1:d.I
    for j in 1:d.J
      for n in 1:d.N[i]
        update_gam!(i, n, j, s, c, d)
      end
    end
  end
end

