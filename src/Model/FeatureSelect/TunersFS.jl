mutable struct TunersFS{T <: Any}
  W_star::Matrix{MCMC.TuningParam{Float64}}
  p::T
  tuners::Tuners

  TunersFS{T}() where T = new()
end


# TODO: Profile this.
function TunersFS(tuners::Tuners, s::State)
  J, K = size(s.Z)
  I = length(s.y_imputed)
  W_star_tuner = [MCMC.TuningParam(1.0) for i in 1:I, k in 1:K]

  # TODO: Profile this. Is this inefficient?
  tfs = TunersFS{Any}()

  tfs.W_star = W_star_tuner
  tfs.tuners = tuners

  return tfs
end
