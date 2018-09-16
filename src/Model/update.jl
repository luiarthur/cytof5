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

function update_state(s::State, c::Constants, d::Data, tuners::Tuners, ll::Vector{Float64})
  # Gibbs.
  update_Z(s, c, d) # Need to check.
  update_mus(s, c, d) # DONE
  update_alpha(s, c, d) # DONE
  update_v(s, c, d) # DONE
  update_W(s, c, d) # DONE
  update_eta(s, c, d) # DONE
  update_lam(s, c, d) # DONE. Need to check.
  update_gam(s, c, d) # DONE
  update_sig2(s, c, d) # DONE

  # Metropolis.
  update_y_imputed(s, c, d, tuners) # DONE
  update_b(s, c, d, tuners) # DONE

  # Compute loglikelihood.
  append!(ll, compute_loglike(s, c, d))
end
