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
  # Note: `@ifTrue` is defined in "util.jl"

  # Return true if sym not is not fixed
  isRandom(sym::Symbol)::Bool = !(sym in fix)

  # Gibbs.
  @ifTrue isRandom(:Z)          update_Z(s, c, d)
  @ifTrue isRandom(:mus)        update_mus(s, c, d)
  @ifTrue isRandom(:alpha)      update_alpha(s, c, d)
  @ifTrue isRandom(:v)          update_v(s, c, d)
  @ifTrue isRandom(:W)          update_W(s, c, d)
  @ifTrue isRandom(:eta)        update_eta(s, c, d)
  @ifTrue isRandom(:lam)        update_lam(s, c, d)
  @ifTrue isRandom(:gam)        update_gam(s, c, d) 
  @ifTrue isRandom(:sig2)       update_sig2(s, c, d) 

  # Metropolis.
  @ifTrue isRandom(:y_imputed)  update_y_imputed(s, c, d, tuners) 
  @ifTrue isRandom(:b0)         update_b0(s, c, d, tuners) 
  @ifTrue isRandom(:b1)         update_b1(s, c, d, tuners) 

  # Compute loglikelihood.
  append!(ll, compute_loglike(s, c, d))
end
