function update_b0(i::Int, s::State, c::Constants, d::Data, tuners::Tuners)
  function ll(b0i::Float64)
    out = 0.0

    # TODO: Can I vectorize this?
    for j in 1:d.J
      #for n in 1:d.N[i]
      #  pinj = prob_miss(s.y_imputed[i][n, j], b0i, s.b1[i])
      #  out += logpdf(Bernoulli(pinj), d.m[i][n, j])
      #end
      pij = prob_miss.(s.y_imputed[i][:, j], b0i, s.b1[i])
      out += sum(logpdf.(Bernoulli.(pij), d.m[i][:, j]))
    end

    return out
  end

  function lp(b0i::Float64)
    return logpdf(c.b0_prior, b0i)
  end

  logFullCond(b0i::Float64) = ll(b0i) + lp(b0i)

  s.b0[i] = MCMC.metropolisAdaptive(s.b0[i], logFullCond, tuners.b0[i])
end

function update_b1(i::Int, s::State, c::Constants, d::Data, tuners::Tuners)

end

function update_b(s::State, c::Constants, d::Data, tuners::Tuners)
  for i in 1:d.I
    update_b0(i, s, c, d, tuners)
    update_b1(i, s, c, d, tuners)
  end
end
