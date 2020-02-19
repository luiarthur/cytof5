#=
using Revise
=#
module HMC

using Flux
using Distributions

"""
Get random momentum from a TrackedArray parameter.
"""
rand_momentum(x::TrackedArray) = randn(size(x))


"""
Compute kinetic energy given momentum
"""
function compute_kinetic_energy(momentum)
  return sum([sum(v .^ 2) for v in momentum]) / 2.0
end


# Given an arbitrary type, get the Flux parameters
function get_params(s::S) where S
  ps = []
  for key in fieldnames(S)
    if isdefined(s, key)
      f = getfield(s, key)
      if Tracker.istracked(f)
        append!(ps, [f])
      end
    else
      ErrorException("Field $key is undefined in $(S.name)")
    end
  end
  return Flux.Params(ps)
end


"""
See p. 14 of
https://arxiv.org/pdf/1206.1901.pdf
"""
function hmc_update(curr_state::S, log_prob::Function,
                    num_leapfrog_steps::Int, eps::Float64;
                    momentum_sd::Real=1) where S
  # TODO: Add an invM argument. See BDA3 p. 301 - 302.

  state = deepcopy(curr_state)

  # Get tracked parameters
  qs = get_params(state)

  # Create a dictionary state -> momentum
  ps = [rand_momentum(q) * momentum_sd for q in qs]

  # Current kinetic energy
  curr_K = compute_kinetic_energy(values(ps))
  
  # Potential enerdy
  U(s) = -log_prob(s)

  # Current potential energy
  curr_U = Tracker.data(U(state))

  # Compute gradient of U
  grad_U(qs) = Tracker.gradient(() -> U(state), qs)

  # Make a half step for momentum at the beginning
  gs = grad_U(qs)
  for (q, p) in zip(qs, ps)
    # p .-= eps * Tracker.data(gs[q]) / 2.0
    Flux.Tracker.update!(p, -eps * gs[q] / 2)
  end

  # Alternate full steps for position and momentum
  for i in 1:num_leapfrog_steps
    # Make a full step for the position
    for (q, p) in zip(qs, ps)
      Flux.Tracker.update!(q, eps * p)
    end

    # Make a full step for the momentum, except at end of trajectory
    if i < num_leapfrog_steps
      gs = grad_U(qs)
      for (q, p) in zip(qs, ps)
        # p .-= eps * Tracker.data(gs[q])
        Flux.Tracker.update!(p, -eps * gs[q])
      end
    end
  end

  # Make a half step for momentum at the end
  gs = grad_U(qs)
  for (q, p) in zip(qs, ps)
    # p .-= eps * Tracker.data(gs[q]) / 2.0 
    Flux.Tracker.update!(p, -eps * gs[q] / 2)
  end

  # Proposed potential energy
  cand_U = Tracker.data(U(state))

  # Proposed kinetic  energy
  cand_K = Tracker.data(compute_kinetic_energy(values(ps)))

  ### Metropolis step ###
  # log acceptance ratio.  NOTE: potential = -log_prob
  log_acceptance_ratio = curr_U + curr_K - cand_U - cand_K

  # Accept or reject
  if log_acceptance_ratio > log(rand())
    return state, -cand_U
  else
    return curr_state, -curr_U
  end
  ### End of Metropolis step ###
end

end # module HMC
