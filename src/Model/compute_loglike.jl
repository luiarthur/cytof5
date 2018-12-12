function compute_like(i::Int, n::Int, j::Int, s::State, c::Constants, d::Data)::Float64
  p = prob_miss(s.y_imputed[i][n, j], c.beta[:, i])
  like = 1.0
  y_inj_is_missing = (d.m[i][n, j] == 1)

  if y_inj_is_missing
    # Don't compute this if it's observed because missing mechanism is fixed
    # because this will be a constant 
    like *= pdf(Bernoulli(p), d.m[i][n, j])
  else
    # multiply to likelihood for y_observed (non-missing)
    k = s.lam[i][n]
    if k > 0
      z = s.Z[j, k]
      l = s.gam[i][n, j]
      like *= pdf(Normal(s.mus[z][l], sqrt(s.sig2[i])), d.y[i][n, j])
    else
      like *= exp(logdnoisy(i, n, s, c, d))
    end
  end

  if iszero(like)
    println("WARNING: like = 0 for (i: $i, n: $n, j: $j).")
  end

  return like
end


function compute_loglike(i::Int, n::Int, j::Int, s::State, c::Constants, d::Data)::Float64
  p = prob_miss(s.y_imputed[i][n, j], c.beta[:, i])
  ll = 0.0
  y_inj_is_missing = (d.m[i][n, j] == 1)

  if y_inj_is_missing
    # Don't compute this if it's observed because missing mechanism is fixed
    # because this will be a constant 
    ll += logpdf(Bernoulli(p), d.m[i][n, j])
  else
    # Add to log-likelihood for y_observed (non-missing)
    k = s.lam[i][n]
    if k > 0
      z = s.Z[j, k]
      l = s.gam[i][n, j]
      ll += logpdf(Normal(s.mus[z][l], sqrt(s.sig2[i])), d.y[i][n, j])
    else
      ll += logdnoisy(i, n, s, c, d)
    end
  end

  if isinf(ll)
    println("WARNING: loglike = -Inf for (i: $i, n: $n, j: $j).")
  end

  return ll
end


function compute_loglike(s::State, c::Constants, d::Data; normalize::Bool=true)::Float64
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

