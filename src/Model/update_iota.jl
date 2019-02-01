# TODO: Test
function update_iota(s::State, c::Constants, d::Data, tuners::Tuners)
  function lp(iota::Float64)::Float64
    return logpdf(c.iota_prior, iota)
  end

  function ll(iota::Float64)::Float64
    (m0, s0, lower0, _) = params(c.mus_prior[0])
    (m1, s1, _, upper1) = params(c.mus_prior[1])
    
    # OLD
    # return -logcdf(Normal(m0, s0), -iota) - logccdf(Normal(m1, s1), iota)
    # TODO: RUN CB TEST
    out = 0.0
    out -= MCMC.logsumexp(logcdf(Normal(m0, s0), -iota),
                          logccdf(Normal(m0, s0), lower0))
    out -= MCMC.logsumexp(logcdf(Normal(m1, s1), upper1),
                          logccdf(Normal(m1, s1), iota))
  end

  upper = min(-s.mus[0][end], s.mus[1][1])
  s.iota = MCMC.metLogitAdaptive(s.iota, ll, lp, tuners.iota, a=0.0, b=upper)
  # println(s.iota)
  # println(s.mus)
end

