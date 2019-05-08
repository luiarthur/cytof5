function compute_elbo(state::StateMP, y::Vector{M}, c::Constants,
                      metrics::Dict{Symbol, Vector{Float64}}) where M
  real, tran, yout, log_qy = rsample(state, y, c);

  m = [isnan.(yi) for yi in y]
  ll = loglike(tran, yout, m, c)
  lp = logprior(real, tran, state, c)
  lq = logq(real, state) + log_qy
  elbo = ll + lp - lq

  # store metrics
  append!(metrics[:ll], ll.data)
  append!(metrics[:lp], lp.data)
  append!(metrics[:lq], lq.data)
  append!(metrics[:elbo], elbo.data)

  return elbo
end

