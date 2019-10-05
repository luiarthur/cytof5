@namedargs struct Tuners
  y_imputed::Dict{Tuple{Int64, Int64, Int64}, MCMC.TuningParam{Float64}}
  Z::MCMC.TuningParam{Float64}
  v::Vector{MCMC.TuningParam{Float64}}
end


# TODO
# Assert this is correct!
function Tuners(y, K; z_prob_flip=nothing)
  I = length(y)
  J = ncol(y[1])
  N = nrow.(y)

  y_tuner = Dict{Tuple{Int, Int, Int}, MCMC.TuningParam{Float64}}()
  for i in 1:I
    for n in 1:N[i]
      for j in 1:J
        if isnan(y[i][n, j])
          y_tuner[i, n, j] = MCMC.TuningParam(1.0)
        end
      end
    end
  end

  if z_prob_flip == nothing
    z_prob_flip = 1. / (J * K)
  end

  Z_tuner = MCMC.TuningParam(z_prob_flip)
  v_tuner = [MCMC.TuningParam(1.0) for k in 1:K]

  return Tuners(y_imputed=y_tuner, Z=Z_tuner, v=v_tuner)
end
