const Cube = Array{T, 3} where T

@namedargs mutable struct State{F <: AbstractFloat}
  Z::Matrix{Bool} # Dim: J x K. Z[j,k] ∈ {0, 1}. true => 1, false => 0
  delta::Dict{Bool, Vector{F}} # delta_z dim: L[z]
  alpha::F
  v::Vector{F} # dim: K
  W::Matrix{F} # dim: I x K
  sig2::Vector{F} # dim: I
  eta::Dict{Bool, Cube{F}} # eta_zik dim: L[z]
  lam::Vector{Vector{Int8}} # Array of Array. lam[1:I] ∈ {1,...,K}
  gam::Vector{Matrix{Int8}} # gam_ij dim: N[i]
  y_imputed::Vector{Matrix{F}} # y_ij dim: N[i]
  eps::Vector{F} # dim I
end

#= Note:
genInitialState(c::Constants, d::Data)
is in Constants.jl.
=#

function compress(state::State)
  warn_msg = "WARNING: The `compress` function has not been fully tested "
  warn_msg *= "and may result in errors!"
  @warn warn_msg

  if typeof(state) == State{Float32}
    return state
  else
    return State(Z=Matrix{Bool}(state.Z),
                 delta=Dict{Bool, Vector{Float32}}(state.delta),
                 alpha=Float32(state.alpha),
                 v=Float32.(state.v),
                 W=Float32.(state.W),
                 sig2=Float32.(state.sig2),
                 eta=Dict{Bool, Cube{Float32}}(state.eta),
                 lam=Vector{Vector{Int8}}(state.lam),
                 gam=Vector{Matrix{Int8}}(state.gam),
                 eps=Float32.(state.eps),
                 y_imputed=Vector{Matrix{Float32}}(state.y_imputed))
  end
end
