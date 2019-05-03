function logq(real::State{F, A1, A2, A3}, mps, c::Constants{E}) where {E, F, A1, A2, A3}
  lq = zero(F)

  for key in fieldnames(State)
    if !(key in (:y_m, :y_log_s))
      mp = getfield(mps, key)
      r = getfield(real, key)

      lq += sum(ADVI.log_q(mp, r))
    end
  end

  return lq
end


