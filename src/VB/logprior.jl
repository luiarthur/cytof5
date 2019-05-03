function logprior(real::State{F, A1, A2, A3},
                  tran::State{F, A1, A2, A3},
                  mps, #::StateMP{F},
                  c::Constants{E}) where {E, F, A1, A2, A3}

  lp = zero(F)
  for key in fieldnames(State)
    if !(key in (:y_m, :y_log_s))
      if key == :v
        if c.use_stickbreak
          lp += sum(logpdf.(Beta(tran.alpha, 1), tran.v))
        else
          lp += sum(logpdf.(Beta(tran.alpha / c.K, 1), tran.v))
        end
        lp += sum(ADVI.logabsdetJ(mps.v, real.v, tran.v))
      else # key != :v
        p = getfield(c.priors, key)
        t = getfield(tran, key)
        r = getfield(real, key)
        mp = getfield(mps, key)

        if isa(p, Dirichlet) # W, eta
          lpdf = ADVI.lpdf_dirichlet(p.alpha, t)
        else
          lpdf = logpdf.(p, t)
        end
        labsdj = ADVI.logabsdetJ(mp, r, t)
        lp += sum(lpdf + labsdj)
      end
    end
  end

  return lp
end
