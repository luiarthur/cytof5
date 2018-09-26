function update_mus(s::State, c::Constants, d::Data)
  sumYOverSig2 = Dict([z => zeros(c.L) for z in 0:1])
  cardinality = Dict([z => zeros(Int, d.I, c.L) for z in 0:1])
  for i in 1:d.I
    for j in 1:d.J
      for n in 1:d.N[i]
        l = s.gam[i][n, j]
        z = s.Z[j, s.lam[i][n]]
        sumYOverSig2[z][l] += s.y_imputed[i][n, j] / s.sig2[i]
        cardinality[z][i, l] += 1
      end
    end
  end

  for z in 0:1
    for l in 1:c.L
      priorVar = var(priorMu(z, l, s, c))
      priorMean = mean(priorMu(z, l, s, c))
      newDenom = (1 + priorVar * sum(cardinality[z][:, l] ./ s.sig2))
      if priorVar < 0
        printstyled("WARNING: mus priorVar is negative: $(priorVar)\n", color="yellow")
        priorVar = 1E-10
      end
      newMean = (priorMean + priorVar * sumYOverSig2[z][l]) / newDenom
      newSd = sqrt(priorVar / newDenom)
      newLower = minimum(priorMu(z, l, s, c))
      newUpper = maximum(priorMu(z, l, s, c))
      s.mus[z][l] = rand(TruncatedNormal(newMean, newSd, newLower, newUpper))
    end
  end
end
