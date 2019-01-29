@namedargs struct Tuners
  y_imputed::Dict{Tuple{Int64, Int64, Int64}, MCMC.TuningParam}
  Z::MCMC.TuningParam
  mus::Dict{Bool, Vector{MCMC.TuningParam}}
end
