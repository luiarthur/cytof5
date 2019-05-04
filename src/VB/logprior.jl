function logprior(real::State{A1, A2, A3},
                  tran::State{A1, A2, A3},
                  mps::StateMP,
                  c::Constants) where {A1, A2, A3}

  lp = 0
  for key in fieldnames(State)
    if !(key in (:y_m, :y_log_s))
      if key == :v
        a = tran.alpha[1]
        b = one(a)
        if c.use_stickbreak
          # lp += sum(logpdf.(Beta(a, b), tran.v))
          lp += sum(ADVI.lpdf_beta.(tran.v, a, b))
        else
          lp += sum(ADVI.lpdf_beta.(tran.v, a/c.K, b))
        end
        lp += sum(ADVI.logabsdetJ(mps.v, real.v, tran.v))
      else # key != :v
        p = getfield(c.priors, key)
        t = getfield(tran, key)
        r = getfield(real, key)
        mp = getfield(mps, key)

        # if isa(p, Dirichlet) # W, eta
        #   lpdf = ADVI.lpdf_dirichlet(t, p.alpha)
        # else
        #   # lpdf = logpdf.(p, t)
        # end
        lpdf = ADVI.compute_lpdf(p, t)
        labsdj = ADVI.logabsdetJ(mp, r, t)
        # lp += sum(lpdf + labsdj)
        lp += sum(labsdj)
      end
    end
  end

  return lp
end
