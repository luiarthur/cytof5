@namedargs struct Tuners
  y_imputed::Dict{Tuple{Int64, Int64, Int64}, MCMC.TuningParam{Float64}}
  Z::MCMC.TuningParam{Float64}
  v::Vector{MCMC.TuningParam{Float64}}
end
