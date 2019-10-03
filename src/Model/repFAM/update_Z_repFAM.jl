# TODO

"""
`z` is required to be 0 or 1.
With probability `probFlip`, change z to 1 or 0.
"""
function flip_bit(z::Bool, probFlip::Float64)::Bool
  return probFlip > rand() ? !z : z
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

function update_Z_repFAM(s::State, c::Constants, d::Data, tuners::Tuners,
                         sb_ibp::Bool)
  #cand_Z = flip_bit.(s.Z, MCMC.logit(tuners.Z.value, a=0.0, b=1.0))
  cand_Z = Matrix{Bool}(flip_bit.(s.Z, c.probFlip_Z))

  curr_Z = s.Z

  function log_fc(Z)
    v = sb_ibp ? cumprod(s.v) : s.v
    lp = logprob_Z_repFAM(Z, v, c.similarity_Z)
    ll = 0.0

    for i in 1:d.I
      for j in 1:d.J
        for n in 1:d.N[i]
          k = s.lam[i][n]
          if k > 0
            z = Z[j, k]
            ll += log(dmixture(z, i, n, j, s, c, d))
          end
        end
      end
    end

    return lp + ll
  end

  accept = log_fc(cand_Z) - log_fc(curr_Z) > log(rand())
  if accept
    println("Joint-update of Z resulted in a move.")
    s.Z .= cand_Z
  end

  # Update tuning param
  #MCMC.update_tuning_param_default(tuners.Z, accept)
end
