function dmixture(z::Integer, i::Integer, n::Integer, j::Integer,
                  s::State, c::Constants, d::Data)::Float64
  sd = sqrt(s.sig2[i])
  dvec = s.eta[z][i, j, :] .* pdf.(Normal.(s.mus[z], sd), s.y_imputed[i][n, j])
  return sum(dvec)
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
    out = pdf.(Normal.(s.mus[z][l], sd), c.y_grid)
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


# For updating Z marginalizing over lambda and gamma ############
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

function log_dmix_nolamgam(Z::Matrix{Bool}, s::State, c::Constants, d::Data)::Float64
  out = 0.0

  for i in 1:d.I
    for n in 1:d.N[i]
      out += log(dmix_nolamgam(Z, i, n, s, c, d))
    end
  end

  return out
end
