const Cube = Array{T, 3} where T

@namedargs mutable struct State{F <: AbstractFloat}
  Z::Matrix{Bool} # Dim: J x K. Z[j,k] ∈ {0, 1}. true => 1, false => 0
  mus::Dict{Bool, Vector{F}}
  iota::F
  alpha::F
  v::Vector{F}
  W::Matrix{F}
  sig2::Vector{F}
  eta::Dict{Bool, Cube{F}}
  lam::Vector{Vector{Int8}} # Array of Array. lam[1:I] ∈ {1,...,K}
  gam::Vector{Matrix{Int8}}
  y_imputed::Vector{Matrix{F}}
  eps::Vector{F} # dim I
end

#= Note:
genInitialState(c::Constants, d::Data)
is in Constants.jl.
=#

function compress(state::State)
  println("WARNING: The `compress` function has not been fully tested and will result in errors!")
  if typeof(state) == State{Float32}
    return state
  else
    return State(Z=Matrix{Bool}(state.Z),
                 mus=Dict{Bool, Vector{Float32}}(state.mus),
                 alpha=Float32(state.alpha),
                 v=Float32.(state.v),
                 W=Float32.(state.W),
                 sig2=Float32.(state.sig2),
                 eta=Dict{Bool, Cube{Float32}}(state.eta),
                 lam=Vector{Vector{Int8}}(state.lam),
                 gam=Vector{Matrix{Int8}}(state.gam),
                 eps=Float32.(state.eps),
                 iota=Float32.(state.iota),
                 y_imputed=Vector{Matrix{Float32}}(state.y_imputed))
  end
end
