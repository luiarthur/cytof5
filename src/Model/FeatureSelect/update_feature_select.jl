function update_state_feature_select!(s::StateFS, c::ConstantsFS, d::DataFS,
                                      t::TunersFS, ll::Vector{Float64},
                                      fix::Vector{Symbol},
                                      use_repulsive::Bool,
                                      joint_update_Z::Bool, sb_ibp::Bool)
  # NOTE: `@doIf` is defined in "../util.jl"

  # Return true if parameter (sym) is not fixed
  isRandom(sym::Symbol)::Bool = !(sym in fix)

  # Parameter update-order should be:
  # Z -> v -> alpha -> 
  # omega -> r -> lam -> W* -> gamma -> eta -> delta -> sig2 -> y*

  @doIf isRandom(:Z) if use_repulsive
    update_Z_repFAM!(s.theta, c.constants, d.data, t.tuners, sb_ibp)
  else
    if joint_update_Z
      update_Z_v2!(s.theta, c.constants, d.data, t.tuners, sb_ibp)
    else
      # Do regular updates
      update_Z!(s.theta, c.constants, d.data, sb_ibp)
    end
  end

  @doIf isRandom(:v) update_v!(s.theta, c.constants, d.data, t.tuners, sb_ibp)
  @doIf isRandom(:alpha) update_alpha!(s.theta, c.constants, d.data, sb_ibp)
  @doIf isRandom(:omega) update_omega!(s, c, d, t)
  @doIf isRandom(:r) update_r!(s, c, d)
  @doIf isRandom(:lam) update_lam!(s.theta, c.constants, d.data)
  @doIf isRandom(:W_star) update_W_star!(s, c, d, t)
  @doIf isRandom(:gam) update_gam!(s.theta, c.constants, d.data)
  @doIf isRandom(:eta) update_eta!(s.theta, c.constants, d.data)
  @doIf isRandom(:delta) update_delta!(s.theta, c.constants, d.data)
  @doIf isRandom(:sig2) update_sig2!(s.theta, c.constants, d.data)
  @doIf isRandom(:y_imputed)  update_y_imputed!(s.theta, c.constants, d.data,
                                                t.tuners) 

  # TODO: Compute loglikelihood.
  # append!(ll, compute_loglike(s, c, d))
end

