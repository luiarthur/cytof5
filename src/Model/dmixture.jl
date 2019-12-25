function dmixture(z::Integer, i::Integer, n::Integer, j::Integer,
                  s::State, c::Constants, d::Data)::Float64
  sd = sqrt(s.sig2[i])
  dvec = s.eta[z][i, j, :]
  dvec .*= pdf.(Normal.(mus(Bool(z), s, c, d), sd), s.y_imputed[i][n, j])
  return sum(dvec)
end

function logdmixture(z::Integer, i::Integer, n::Integer, j::Integer,
                     s::State, c::Constants, d::Data)::Float64
  sd = sqrt(s.sig2[i])
  logdvec = log.(s.eta[z][i, j, :])
  logdvec += logpdf.(Normal.(mus(Bool(z), s, c, d), sd), s.y_imputed[i][n, j])
  return MCMC.logsumexp(logdvec)
end


function logdnoisy(i::Integer, n::Integer,
                   s::State, c::Constants, d::Data)::Float64
  return sum(logpdf.(c.noisyDist, s.y_imputed[i][n, :]))
end

function dnoisy(i::Integer, n::Integer, s::State, c::Constants, d::Data)::Float64
  exp(logdnoisy(i, n, s, c, d))
end

function datadensity(i::Integer, n::Integer, j::Integer,
                     s::State, c::Constants, d::Data)::Vector{Float64}
  out = zeros(length(c.y_grid))

  k = s.lam[i][n]
  if k > 0
    z = s.Z[j, k]
    l = s.gam[i][n, j]
    sd = sqrt(s.sig2[i])
    out = pdf.(Normal.(mus(z, l, s, c, d), sd), c.y_grid)
  else
    out = pdf.(c.noisyDist, c.y_grid)
  end

  return out
end

function datadensity(i::Integer, j::Integer,
                     s::State, c::Constants, d::Data)::Vector{Float64}
  n_obs = sum(d.m[i][:, j] .== 0)
  out = zeros(length(c.y_grid))

  for n in 1:d.N[i]
    if d.m[i][n, j] == 0
      out += datadensity(i, n, j, s, c, d) / n_obs
    end
  end

  return out
end


### For updating Z marginalizing over lambda and gamma ############


# TODO: DEPRECATE ###########################################################
function dmix_nolamgam(Z::Matrix{Bool}, i::Integer, n::Integer,
                       s::State, c::Constants, d::Data)::Float64

  # Get the density over all markers
  dyin_not_noisy = 0.0
  for k in 1:c.K
    dvec = prod(dmixture(Z[j, k], i, n, j, s, c, d) for j in 1:d.J)
    dyin_not_noisy += s.W[i, k] * dvec
  end

  return s.eps[i] * dnoisy(i, n, s, c, d) + (1 - s.eps[i]) * dyin_not_noisy
end
#############################################################################


function log_dmix_nolamgam(Z::Matrix{Bool}, i::Integer, n::Integer,
                           s::State, c::Constants, d::Data)::Float64

  # Indices for k such that W[i, k] > 0 (selected)
  selected_k = findall(identity, s.W[i, :] .> 0)

  # Get the density over all markers
  log_dyin_not_noisy = log.(s.W[i, :])

  # NOTE: For the not-selected ones, log(W[i,k]) = -Inf anyway.
  for k in selected_k
    logdvec = sum(logdmixture(Z[j, k], i, n, j, s, c, d) for j in 1:d.J)
    log_dyin_not_noisy[k] += logdvec
  end

  return MCMC.logsumexp([log(s.eps[i]) + logdnoisy(i, n, s, c, d),
                         log(1 - s.eps[i]) +
                         MCMC.logsumexp(log_dyin_not_noisy)])
end


function log_dmix_nolamgam(Z::Matrix{Bool}, s::State, c::Constants, d::Data)::Float64
  out = 0.0

  for i in 1:d.I
    for n in 1:d.N[i]
      # out += log(dmix_nolamgam(Z, i, n, s, c, d))
      out += log_dmix_nolamgam(Z, i, n, s, c, d)
    end
  end

  return out
end

#####################################################################
function log_dmix_nolamgam(Z::Matrix{Bool}, i::Integer, n::Integer,
                           A::Vector{Vector{Float64}},
                           B0::Vector{Matrix{Float64}},
                           B1::Vector{Matrix{Float64}},
                           s::State, c::Constants, d::Data)::Float64

  # Indices for k such that W[i, k] > 0 (selected)
  selected_k = findall(identity, s.W[i, :] .> 0)

  # Get the density over all markers
  log_dyin_not_noisy = log.(s.W[i, :])

  # NOTE: For the not-selected ones, log(W[i,k]) = -Inf anyway.
  for k in 1:selected_k
    logdvec = sum(Z[j, k] == 0 ? B0[i][n, j] : B1[i][n, j] for j in 1:d.J)
    log_dyin_not_noisy[k] += logdvec
  end

  return MCMC.logsumexp([log(s.eps[i]) + A[i][n],
                        log(1 - s.eps[i]) +
                        MCMC.logsumexp(log_dyin_not_noisy)])
end


function log_dmix_nolamgam(Z::Matrix{Bool},
                           A::Vector{Vector{Float64}},
                           B0::Vector{Matrix{Float64}},
                           B1::Vector{Matrix{Float64}},
                           s::State, c::Constants, d::Data)::Float64
  out = 0.0

  for i in 1:d.I
    for n in 1:d.N[i]
      out += log_dmix_nolamgam(Z, i, n, A, B0, B1, s, c, d)
    end
  end

  return out
end
