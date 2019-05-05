function logprior(real::State{A1, A2, A3},
                  tran::State{A1, A2, A3},
                  mps::StateMP,
                  c::Constants) where {A1, A2, A3}

  lp = 0.0
  for key in fieldnames(State)
    if !(key in (:y_m, :y_log_s))
      if key == :v
        a = tran.alpha[1]
        # +=
        lp = lp + sum(ADVI.compute_lpdf(c.priors.v(a), tran.v))
        lp = lp + sum(ADVI.logabsdetJ(mps.v, real.v, tran.v))
      else # key != :v
        p = getfield(c.priors, key)
        t = getfield(tran, key)
        r = getfield(real, key)
        mp = getfield(mps, key)

        lpdf = ADVI.compute_lpdf(p, t)
        labsdj = ADVI.logabsdetJ(mp, r, t)
        # +=
        lp = lp + sum(lpdf + labsdj)
      end
    end
  end

  @assert !isnan(lp)
  @assert !isinf(lp)

  return lp
end
