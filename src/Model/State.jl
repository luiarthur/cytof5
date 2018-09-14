const Cube = Array{T, 3} where T

mutable struct State
  Z::Matrix{Int} # Dim: J x K. Z[j,k] ∈ {0, 1}
  mus::Dict{Int, Matrix{Float64}}
  alpha::Float64
  v::Vector{Float64}
  W::Matrix{Float64}
  sig2::Vector{Float64}
  eta::Dict{Int, Cube{Float64}}
  lam::Vector{Vector{Int16}} # Array of Array. lam[1:I] ∈ {1,...,K}
  gam::Vector{Matrix{Float64}}
  y_imputed::Vector{Matrix{Float64}}
  b0::Float64
  b1::Float64
end

