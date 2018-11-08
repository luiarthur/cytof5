@ann struct Tuners
  b0::Vector{MCMC.TuningParam}
  b1::Vector{MCMC.TuningParam}
  y_imputed::Dict{Tuple{Int64, Int64, Int64}, MCMC.TuningParam}
  Z::MCMC.TuningParam
end
