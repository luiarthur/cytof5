function compute_like(i::Int, n::Int, j::Int, s::State, c::Constants, d::Data)
  p = prob_miss(s.y_imputed[i][n, j], s.b0[i], s.b1[i])
  like = pdf(Bernoulli(p), d.m[i][n, j])
  # For numerical stability. Ensure like > 0.
  # TODO: make EPS=1E-8 an option to be specified
  @leftTrunc! 1E-8 like

  # multiply to likelihood for y_observed (non-missing)
  if d.m[i][n, j] == 0
    z = s.Z[j, s.lam[i][n]]
    l = s.gam[i][n, j]
    like *= pdf(Normal(s.mus[z][l], sqrt(s.sig2[i])), d.y[i][n, j])
  end

  return like
end


function compute_loglike(i::Int, n::Int, j::Int, s::State, c::Constants, d::Data)
  ll = logpdf(Bernoulli(prob_miss(s.y_imputed[i][n, j], s.b0[i], s.b1[i])), d.m[i][n, j])
  # For numerical stability. Ensure ll > -Inf.
  # TODO: make MIN_ll=-1E8 an option to be specified
  @leftTrunc! -1E8 ll

  # Add to likelihood for y_observed (non-missing)
  if d.m[i][n, j] == 0
    z = s.Z[j, s.lam[i][n]]
    l = s.gam[i][n, j]
    ll += logpdf(Normal(s.mus[z][l], sqrt(s.sig2[i])), d.y[i][n, j])
  end

  return ll
end


function compute_loglike(s::State, c::Constants, d::Data; normalize::Bool=true)
  ll = 0.0

  sumN = sum(d.N)

  for i in 1:d.I
    for j in 1:d.J
      for n in 1:d.N[i]
        if normalize
          ll += compute_loglike(i, n, j, s, c, d) / sumN
        else
          ll += compute_loglike(i, n, j, s, c, d)
        end
      end
    end
  end

  return ll
end
