function compute_loglike(i::Int, n::Int, j::Int,
                         s::State, c::Constants, d::Data)::Float64
  ll = 0.0
  y_inj_is_missing = (d.m[i][n, j] == 1)

  if y_inj_is_missing
    # NOTE: Don't compute p(m_inj | y_inj, theta) if y_inj is observed,
    # because missing mechanism is fixed, and will results in a constant.
    
    # Compute p(m_inj | y_inj, theta) term.
    p = prob_miss(s.y_imputed[i][n, j], c.beta[:, i])
    ll += log(p)
  end

  # Compute p(y_inj | theta) term.
  k = s.lam[i][n]
  y_inj = d.y[i][n, j]
  if k > 0  # cell is not noisy 
    z = s.Z[j, k]
    l = s.gam[i][n, j]
    ll += logpdf(Normal(mus(z, l, s, c, d), sqrt(s.sig2[i])), y_inj)
  else  # cell is noisy and observed
    ll += logpdf(c.noisyDist, y_inj)
  end

  if isinf(ll)
    println("WARNING: loglike = -Inf for (i: $i, n: $n, j: $j).")
  end

  return ll
end

function compute_like(i::Int, n::Int, j::Int, s::State, c::Constants, d::Data)::Float64
  return exp(compute_loglike(i, n, j, s, c, d))
end

function compute_like(i::Int, n::Int, s::State, c::Constants, d::Data)::Float64
  ll_in = 0.0

  for j in 1:d.J
    ll_in += compute_loglike(i, n, j, s, c, d)
  end

  return exp(ll_in)
end


function compute_loglike(s::State, c::Constants, d::Data;
                         normalize::Bool=true)::Float64
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
