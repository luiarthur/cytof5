include("dmixture.jl")
include("update_Z.jl")
include("update_mus.jl")
include("update_alpha.jl")
include("update_v.jl")
include("update_W.jl")
include("update_sig2.jl")
include("update_eta.jl")
include("update_lam.jl")
include("update_gam.jl")
include("update_y_imputed.jl")
include("update_b.jl")
include("compute_loglike.jl") # TODO

function update_state(s::State, c::Constants, d::Data, tuners::Tuners,
                      ll::Vector{Float64}, fix::Vector{Symbol})
  # Gibbs.
  if !(:Z in fix) update_Z(s, c, d) end
  if !(:mus in fix) update_mus(s, c, d) end
  if !(:alpha in fix) update_alpha(s, c, d) end
    if !(:v in fix) update_v(s, c, d) end
  if !(:W in fix) update_W(s, c, d) end
  if !(:eta in fix) update_eta(s, c, d) end
  if !(:lam in fix) update_lam(s, c, d) end
  if !(:gam in fix) update_gam(s, c, d) end
  if !(:sig2 in fix) update_sig2(s, c, d) end

  # Metropolis.
  if !(:y_imputed in fix) update_y_imputed(s, c, d, tuners) end
  if !(:b0 in fix) update_b0(s, c, d, tuners) end
  if !(:b1 in fix) update_b1(s, c, d, tuners) end

  # Compute loglikelihood.
  append!(ll, compute_loglike(s, c, d))
end
