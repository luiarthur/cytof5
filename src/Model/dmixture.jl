function dmixture(z::Int, i::Int, n::Int, j::Int, s::State, c::Constants, d::Data)
  sd = sqrt(s.sig2[i])
  dvec = s.eta[z][i, j, :] .* pdf.(Normal.(s.mus[z], sd), s.y_imputed[i][n, j])
  return sum(dvec)
end

