const Cube = Array{T, 3} where T

mutable struct State{F <: AbstractFloat}
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
  State{F}() where {F <: AbstractFloat} = new()
end

#= Note:
genInitialState(c::Constants, d::Data)
is in Constants.jl.
=#

function compress(state::State)
  warn_msg = "WARNING: The `compress` function has not been fully tested "
  warn_msg *= "and may result in errors!"
  println(warn_msg)

  if typeof(state) == State{Float32}
    return state
  else
    new_state = State{Float32}()
    new_state.Z = Matrix{Bool}(state.Z)
    new_state.delta = Dict{Bool, Vector{Float32}}(state.delta)
    new_state.alpha = Float32(state.alpha)
    new_state.v = Float32.(state.v)
    new_state.W = Float32.(state.W)
    new_state.sig2 = Float32.(state.sig2)
    new_state.eta = Dict{Bool, Cube{Float32}}(state.eta)
    new_state.lam = Vector{Vector{Int8}}(state.lam)
    new_state.gam = Vector{Matrix{Int8}}(state.gam)
    new_state.eps = Float32.(state.eps)
    new_state.y_imputed = Vector{Matrix{Float32}}(state.y_imputed)
    return new_state
  end
end
