function logq(real::State{F, A1, A2, A3}, mps) where {F, A1, A2, A3}
  lq = 0

  for key in fieldnames(State)
    if !(key in (:y_m, :y_log_s))
      mp = getfield(mps, key)
      r = getfield(real, key)

      lq += sum(ADVI.log_q(mp, r))
    end
  end

  return lq
end


