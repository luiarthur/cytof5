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
  lp = sum(Z * log.(v) + (1 .- Z) * log.(1 .- v))

  # Repulsive component
  for k1 in 1:(K-1)
    for k2 in (k1+1):K
      lp += log(1 - similarity(Z[:, k1], Z[:, k2]))
    end
  end

  return lp
end


function logfullcond_Z_repfam(Z::Matrix{Bool}, s::State, c::Constants,
                                d::Data, t::Tuners, sb_ibp::Bool)
  v = sb_ibp ? cumprod(s.v) : s.v

  # Log prior
  lp = logprob_Z_repFAM(Z, v, c.similarity_Z)

  # log likelihood
  ll = 0.0

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


function update_Z_repFAM!(s::State, c::Constants, d::Data, t::Tuners,
                          sb_ibp::Bool; debug::Bool=false)
  # cand_Z = flip_bit.(s.Z, MCMC.logit(t.Z.value, a=0.0, b=1.0))
  # cand_Z = Matrix{Bool}(flip_bit.(s.Z, c.probFlip_Z))

  num_bits_to_flip = round(Int, c.probFlip_Z * d.J * c.K)
  num_bits_to_flip = clamp(num_bits_to_flip, 1, d.J * c.K)
  cand_Z = flip_bits(s.Z, num_bits_to_flip)

  update_Z_repFAM!(cand_Z, s, c, d, t, sb_ibp, debug=debug)
end


# function update_Z_repFAM!(s::State, c::Constants, d::Data, t::Tuners,
#                           sb_ibp::Bool; config::Any=nothing)
#   
# end
