function update_lam_logpostvec(i::Int, n::Int, s::State, c::Constants, d::Data)
  logprior0 = log(s.eps[i])
  loglike0 = logdnoisy(i, n, s, c, d)
  logPost0 = logprior0 + loglike0

  logpriorVec = log.(s.W[i,:]) .+ log(1.0 - s.eps[i])
  loglikeVec = zeros(c.K)

  for k in 1:c.K
    for j in 1:d.J
      z = s.Z[j, k]
      loglikeVec[k] += log(dmixture(z, i, n, j, s, c, d))
    end
  end

  logPostVec = logpriorVec .+ loglikeVec
  append!(logPostVec, logPost0)

  return logPostVec
end

function update_lam(i::Int, n::Int, s::State, c::Constants, d::Data)
  logPostVec = update_lam_logpostvec(i, n, s, c, d)
  k = MCMC.wsample_logprob(logPostVec)
  s.lam[i][n] = (k <= c.K ? k : 0)
end

function update_lam(s::State, c::Constants, d::Data)
  for i in 1:d.I
    for n in 1:d.N[i]
      update_lam(i, n, s, c, d)
    end
  end
end
