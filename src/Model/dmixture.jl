function dmixture(z::Integer, i::Integer, n::Integer, j::Integer,
                  s::State, c::Constants, d::Data)::Float64
  sd = sqrt(s.sig2[i])
  dvec = s.eta[z][i, j, :] .* pdf.(Normal.(s.mus[z], sd), s.y_imputed[i][n, j])
  return sum(dvec)
end

function logdnoisy(i::Integer, n::Integer,
                s::State, c::Constants, d::Data)::Float64
  sd = sqrt(c.sig2_0)
  return sum(logpdf.(Normal(0, sd), s.y_imputed[i][n, :]))
end

