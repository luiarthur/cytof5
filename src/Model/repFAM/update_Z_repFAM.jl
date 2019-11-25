"""
`z` is required to be 0 or 1.
With probability `probFlip`, change z to 1 or 0.
"""
function flip_bit(z::Bool, probFlip::Float64)::Bool
  return probFlip > rand() ? !z : z
end


"""
`Z` is required to be AbstractArray{Bool}.
Flip a random selection of `num_bits` bits.
The resulting `Z` should be different from the original `Z` by 
`num_bits` bits.
"""
function flip_bits(Z::AbstractArray{Bool}, num_bits::Integer)
  idx = CartesianIndices(Z)
  idx_to_flip = Distributions.sample(idx, num_bits, replace=false)
  Z_new = deepcopy(Z)
  Z_new[idx_to_flip] .= .!Z_new[idx_to_flip]
  return Z_new
end


"""
default similarity function used in computing log probability of Z_repFAM.
= sum(abs.(z1 .- z2))
"""
function similarity_default(z1::Vector{Bool}, z2::Vector{Bool})::Float64
  exp(-sum(abs.(z1 .- z2)))
end


"""
log penalty term as related to column k only.
"""
function log_penalty_repFAM(k::Integer, Z::Matrix{Bool}, similarity::Function)
  K = size(Z, 2)

  log_penalty = 0.0

  for q in 1:K
    if q != k
      log_penalty += MCMC.log1m(similarity(Z[:, k], Z[:, q]))
    end
  end

  return log_penalty
end


"""
log penalty term in repFAM.
"""
function log_penalty_repFAM(Z::Matrix{Bool}, similarity::Function)::Float64
  K = size(Z, 2)

  log_penalty = 0.0

  for k1 in 1:(K - 1)
    for k2 in (k1 + 1):K
      log_penalty += MCMC.log1m(similarity(Z[:, k1], Z[:, k2]))
    end
  end

  return log_penalty
end


"""
log probability of Z ~ repFam_K(v, C), WITHOUT NORMALIZING CONSTANT
where v ~ Beta(a/K, 1),
and similarity(z_{k1}, z_{k2}) computes the similarity of binary vectors z_{k1}
and z_{k2}.

We require similarity(⋅, ⋅) ∈ (0, 1). And when z_{k1} == z_{k2} exactly,
similarity = 1. Similarly, when distance between z_{k1} and z_{k2} approaches
∞, similarity = 0.
"""
function logprob_Z_repFAM(Z::Matrix{Bool}, v::Vector{Float64},
                          similarity::Function)::Float64
  J, K = size(Z)

  # IBP component
  lp = sum(Z * log.(v) + (1 .- Z) * MCMC.log1m.( v))

  # Repulsive component
  log_penalty = log_penalty_repFAM(Z, similarity)

  return lp + log_penalty
end


function logfullcond_Z_repfam(Z::Matrix{Bool}, s::State, c::Constants,
                              d::Data, t::Tuners, sb_ibp::Bool)
  v = sb_ibp ? cumprod(s.v) : s.v

  # Log prior
  lp = logprob_Z_repFAM(Z, v, c.similarity_Z)

  # log likelihood
  ll = 0.0

  if lp > -Inf
    for i in 1:d.I
      for j in 1:d.J
        for n in 1:d.N[i]
          # Cell type
          k = s.lam[i][n]
          if k > 0  # if not noisy cell type
            z = Z[j, k]
            ll += log(dmixture(z, i, n, j, s, c, d))
          end
        end
      end
    end
  end

  return lp + ll
end


function update_Z_repFAM!(cand_Z::Matrix{Bool}, s::State, c::Constants,
                          d::Data, t::Tuners, sb_ibp::Bool; debug::Bool=false)
  # Current Z
  curr_Z = s.Z

  # Log full conditional for candidate Z and current Z
  log_fc_cand = logfullcond_Z_repfam(cand_Z, s, c, d, t, sb_ibp)
  log_fc_curr = logfullcond_Z_repfam(curr_Z, s, c, d, t, sb_ibp)

  # Whether or not to accept proposed Z
  accept = log_fc_cand - log_fc_curr > log(rand())

  if accept
    if debug
      println("Joint-update of Z resulted in a move.")
    end
    s.Z .= cand_Z
  end

  # Update tuning param
  # MCMC.update_tuning_param_default(t.Z, accept)
end


function update_Z_repFAM!(j::Integer, k::Integer,
                          s::State, c::Constants, d::Data, t::Tuners,
                          sb_ibp::Bool; debug::Bool=false)
  # Gibbs:
  Z0 = deepcopy(s.Z)
  Z0[j, k] = false

  Z1 = deepcopy(s.Z)
  Z1[j, k] = true

  log_prob_0 = logfullcond_Z_repfam(Z0, s, c, d, t, sb_ibp)
  log_prob_1 = logfullcond_Z_repfam(Z1, s, c, d, t, sb_ibp)

  p1 = 1 / (1 + exp(log_prob_0 - log_prob_1))

  s.Z[j, k] = p1 > rand()


  # Or Metropolis?
  # cand_Z = deepcopy(s.Z)
  # cand_Z[j, k] = !s.Z[j, k]
  # update_Z_repFAM!(cand_Z, s, c, d, t, sb_ibp, debug=debug)
end


function update_Z_repFAM!(s::State, c::Constants, d::Data, t::Tuners,
                          sb_ibp::Bool; debug::Bool=false)
  if c.probFlip_Z == 1  # Update each Z[j, k] sequentially, instead of jointly.
    for j in 1:d.J
      for k in 1:c.K
        update_Z_repFAM!(j, k, s, c, d, t, sb_ibp, debug=debug)
      end
    end
    # Update a bunch at a time
    cand_Z = flip_bits(s.Z, 3)
    update_Z_repFAM!(cand_Z, s, c, d, t, sb_ibp, debug=debug)
  else  # Flip only some bits to obtain proposed Z.
    num_bits_to_flip = round(Int, c.probFlip_Z * d.J * c.K)
    num_bits_to_flip = clamp(num_bits_to_flip, 1, d.J * c.K)

    # cand_Z = flip_bit.(s.Z, MCMC.logit(t.Z.value, a=0.0, b=1.0))
    # cand_Z = Matrix{Bool}(flip_bit.(s.Z, c.probFlip_Z))
    cand_Z = flip_bits(s.Z, num_bits_to_flip)
    update_Z_repFAM!(cand_Z, s, c, d, t, sb_ibp, debug=debug)
  end
end


# function update_Z_repFAM!(s::State, c::Constants, d::Data, t::Tuners,
#                           sb_ibp::Bool; config::Any=nothing)
#   
# end

#= TODO:
Let R_t = number of rows to update at MCMC iteration t. 
Require that
- P(R_1 = J) = 0.95
- P(R_∞ = j) = .05
- R_t ~ Bin(J - 1; exp(-(t - 1) / 1000)) + 1

1. Row-by-row updates for R_t rows, updating multiple elements at a time.
2. Row-by-row updates, updating one element at at a time.
=#
