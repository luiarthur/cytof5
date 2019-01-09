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
