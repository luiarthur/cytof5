function update_gam(i::Int, n::Int, j::Int, s::State, c::Constants, d::Data)
  if lam[i][n] > 0 s.lam[i][n]
    z = s.Z[j, s.lam[i][n]]
    logpriorVec = log.(s.eta[z][i, j, :])
    loglikeVec = logpdf.(Normal.(s.mus[z], sqrt(s.sig2[i])), s.y_imputed[i][n, j])
    logPostVec = logpriorVec .+ loglikeVec
    s.gam[i][n, j] = MCMC.wsample_logprob(logPostVec)
  else
    s.gam[i][n, j] = 0
  end
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

