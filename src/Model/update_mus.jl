function update_mus(s::State, c::Constants, d::Data)
  sumYOverSig2 = Dict(z => zeros(c.L[z]) for z in 0:1)
  cardinality = Dict(z => zeros(Int, d.I, c.L[z]) for z in 0:1)
  for i in 1:d.I
    for j in 1:d.J
      for n in 1:d.N[i]
        if s.lam[i][n] > 0
          l = s.gam[i][n, j]
          z = s.Z[j, s.lam[i][n]]
          sumYOverSig2[z][l] += s.y_imputed[i][n, j] / s.sig2[i]
          cardinality[z][i, l] += 1
        end
      end
    end
  end

  for z in 0:1
    for l in 1:c.L[z]
      # Note that priorMu and priorSig are NOT the prior mean and std. They are PARAMETERS in 
      # the truncated normal!
      (priorM, priorS, newLower, newUpper) = params(priorMu(z, l, s, c))
      priorS2 = priorS ^ 2
      #priorVar = var(priorMu(z, l, s, c))
      #priorMean = mean(priorMu(z, l, s, c))
      newDenom = (1 + priorS2 * sum(cardinality[z][:, l] ./ s.sig2))
      #if priorVar < 0
      #  printstyled("WARNING: mus priorVar is negative: $(priorVar)\n", color=:yellow)
      #  priorVar = 1E-10
      #end
      newM = (priorM + priorS2 * sumYOverSig2[z][l]) / newDenom
      newS = sqrt(priorS2 / newDenom)
      #newLower = minimum(priorMu(z, l, s, c))
      #newUpper = maximum(priorMu(z, l, s, c))
      s.mus[z][l] = rand(TruncatedNormal(newM, newS, newLower, newUpper))
    end
  end
end
