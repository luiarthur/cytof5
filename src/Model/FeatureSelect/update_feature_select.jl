function update_state_feature_select!(s::StateFS, c::ConstantsFS, d::DataFS,
                                      t::TunersFS;
                                      ll::Vector{Float64},
                                      fix::Vector{Symbol},
                                      use_repulsive::Bool,
                                      joint_update_Z::Bool, sb_ibp::Bool, 
                                      time_updates::Bool=false,
                                      r_marg_lam_freq::Float64=1.0,
                                      Z_thin::Int=1,
                                      joint_update_r::Bool=true)

  # NOTE: `@doIf` is defined in "../util.jl"

  # Return true if parameter (sym) is not fixed
  isRandom(sym::Symbol)::Bool = !(sym in fix)

  function update_Z_!()
    for i in 1:Z_thin
      if joint_update_Z  # marginalize over lambda and gamma occasionally
        update_Z_v2!(s.theta, c.constants, d.data, t.tuners, sb_ibp,
                     use_repulsive=use_repulsive)
      else  # Do regular updates
        update_Z!(s.theta, c.constants, d.data, sb_ibp,
                  use_repulsive=use_repulsive)
      end
    end
  end

  r_marg_lam = (r_marg_lam_freq > rand())

  # Parameter update-order should be:
  # Z -> v -> alpha -> 
  # omega -> r -> lam -> W* -> gamma -> eta -> delta -> sig2 -> y*
  gibbs_sampler = [
    (:Z, () -> update_Z_!()),
    (:v, () -> update_v!(s.theta, c.constants, d.data, t.tuners, sb_ibp)),
    (:alpha, () -> update_alpha!(s.theta, c.constants, d.data, sb_ibp)), 
    (:omega, () -> update_omega!(s, c, d, t)),
    (:r, () -> update_r!(s, c, d, marg_lam=r_marg_lam,
                         joint_update=joint_update_r)),
    (:lam, () -> update_lam!(s.theta, c.constants, d.data)),
    (:W_star, () -> update_W_star!(s, c, d, t)),
    (:gam, () -> update_gam!(s.theta, c.constants, d.data)),
    (:eta, () -> update_eta!(s.theta, c.constants, d.data)),
    (:delta, () -> update_delta!(s.theta, c.constants, d.data)),
    (:sig2, () -> update_sig2!(s.theta, c.constants, d.data)),
    (:y_imputed, () -> update_y_imputed!(s.theta, c.constants, d.data, t.tuners)),
    # Compute loglikelihood.
    (:loglike, () -> append!(ll, compute_loglike(s.theta, c.constants, d.data)))
  ]

  @doIf time_updates println()
  for (param, update_param) in gibbs_sampler
    if isRandom(param)
      tic = time()
      update_param()
      toc = time()

      if time_updates
        println("update time for $(param): $(toc - tic)")
      end
    end
  end

  if time_updates
    println(s.theta.W)
  end
end
