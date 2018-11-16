function dmixture(z::Integer, i::Integer, n::Integer, j::Integer,
                  s::State, c::Constants, d::Data)::Float64
  sd = sqrt(s.sig2[i])
  dvec = s.eta[z][i, j, :] .* pdf.(Normal.(s.mus[z], sd), s.y_imputed[i][n, j])
  return sum(dvec)
end

