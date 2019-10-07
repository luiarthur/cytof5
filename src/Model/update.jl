include("dmixture.jl")
include("update_Z.jl")
include("repFAM/update_Z_repFAM.jl")
include("update_Z_v2.jl")
include("update_delta.jl")
include("update_alpha.jl")
include("update_v.jl")
include("update_W.jl")
include("update_sig2.jl")
include("update_eta.jl")
include("update_lam.jl")
include("update_eps.jl")
include("update_gam.jl")
include("update_y_imputed.jl")
include("compute_loglike.jl")

function update_state!(s::State, c::Constants, d::Data, tuners::Tuners,
                       ll::Vector{Float64}, fix::Vector{Symbol},
                       use_repulsive::Bool, joint_update_Z::Bool, sb_ibp::Bool)

  # NOTE: `@doIf` is defined in "util.jl"

  # Return true if parameter (sym) is not fixed
  isRandom(sym::Symbol)::Bool = !(sym in fix)

  # Gibbs.
  @doIf isRandom(:Z) if use_repulsive
    update_Z_repFAM!(s, c, d, tuners, sb_ibp)
  else
    if joint_update_Z
      update_Z_v2!(s, c, d, tuners, sb_ibp)
    else
      # Do regular updates
      update_Z!(s, c, d, sb_ibp)
    end
  end

  @doIf isRandom(:v)          update_v!(s, c, d, tuners, sb_ibp)
  @doIf isRandom(:alpha)      update_alpha!(s, c, d, sb_ibp)
  @doIf isRandom(:lam)        update_lam!(s, c, d)
  @doIf isRandom(:W)          update_W!(s, c, d)
  @doIf isRandom(:eps)        update_eps!(s, c, d) 

  # gam update must be done after updating Z and before updating delta
  @doIf isRandom(:gam)        update_gam!(s, c, d)
  @doIf isRandom(:eta)        update_eta!(s, c, d)

  @doIf isRandom(:delta)      update_delta!(s, c, d) 
  @doIf isRandom(:sig2)       update_sig2!(s, c, d) 

  # Metropolis.
  @doIf isRandom(:y_imputed)  update_y_imputed!(s, c, d, tuners) 

  # Compute loglikelihood.
  append!(ll, compute_loglike(s, c, d))
end
