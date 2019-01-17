include("dmixture.jl")
include("update_Z.jl")
include("update_Z_repulsive.jl")
include("update_Z_v2.jl")
include("update_mus.jl")
include("update_alpha.jl")
include("update_v.jl")
include("update_W.jl")
include("update_sig2.jl")
include("update_eta.jl")
include("update_lam.jl")
include("update_eps.jl")
include("update_gam.jl")
include("update_y_imputed.jl")
include("compute_loglike.jl") # TODO

function update_state(s::State, c::Constants, d::Data, tuners::Tuners,
                      ll::Vector{Float64}, fix::Vector{Symbol},
                      use_repulsive::Bool, joint_update_Z::Bool)
  # Note: `@doIf` is defined in "util.jl"

  # Return true if sym not is not fixed
  isRandom(sym::Symbol)::Bool = !(sym in fix)

  # Gibbs.
  @doIf isRandom(:Z) if use_repulsive
    update_Z_repulsive(s, c, d, tuners)
  else
    if joint_update_Z
      update_Z_v2(s, c, d, tuners)
    else
      # Do regular updates
      update_Z(s, c, d)
    end
  end

  @doIf isRandom(:v)          update_v(s, c, d)
  @doIf isRandom(:alpha)      update_alpha(s, c, d)
  @doIf isRandom(:lam)        update_lam(s, c, d)
  @doIf isRandom(:W)          update_W(s, c, d)
  @doIf isRandom(:eps)        update_eps(s, c, d) 

  @doIf isRandom(:gam)        update_gam(s, c, d) # must be done between updating Z and mus
  @doIf isRandom(:eta)        update_eta(s, c, d)

  @doIf isRandom(:mus)        update_mus(s, c, d)
  @doIf isRandom(:sig2)       update_sig2(s, c, d) 

  # Metropolis.
  @doIf isRandom(:y_imputed)  update_y_imputed(s, c, d, tuners) 

  # Compute loglikelihood.
  append!(ll, compute_loglike(s, c, d))
end
