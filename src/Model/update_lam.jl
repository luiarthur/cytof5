# TODO: Check if this is correct! I'm not doing mixture here.
function update_lam(i::Int, n::Int, s::State, c::Constants, d::Data)
  logpriorVec = log.(s.W[i,:])
  loglikeVec = zeros(c.K)
  for k in 1:c.K
    for j in 1:d.J
      z = s.Z[j, k]
      loglikeVec[k] += log(dmixture(z, i, n, j, s, c, d))
    end
  end
  logPostVec = logpriorVec .+ loglikeVec
  s.lam[i][n] = MCMC.wsample_logprob(logPostVec)
end

function update_lam(s::State, c::Constants, d::Data)
  for i in 1:d.I
    for n in 1:d.N[i]
      update_lam(i, n, s, c, d)
    end
  end
end
