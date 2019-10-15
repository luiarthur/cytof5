mutable struct TunersFS
  W_star::Matrix{MCMC.TuningParam{Float64}}
  omega::Vector{MCMC.TuningParam{Float64}}
  tuners::Tuners

  TunersFS() = new()
end


function TunersFS(tuners::Tuners, s::State, X::Matrix{Float64})
  J, K = size(s.Z)
  I = length(s.y_imputed)
  W_star_tuner = [MCMC.TuningParam(1.0) for i in 1:I, k in 1:K]

  # number of covariates
  P = size(X, 2)
  omega_tuner = [MCMC.TuningParam(1.0) for p in 1:P]

  tfs = TunersFS()
  tfs.W_star = W_star_tuner
  tfs.omega = omega_tuner
  tfs.tuners = tuners

  return tfs
end
