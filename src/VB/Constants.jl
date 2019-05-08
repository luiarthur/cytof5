# Model Constants
struct Constants
  I::Int
  N::Vector{Int}
  J::Int
  K::Int
  L::Dict{Bool, Int}
  tau::Float64
  beta::Vector{Vector{Float64}} # P x I
  use_stickbreak::Bool
  noisy_var::Float64
  priors::Priors
end

# TODO
function Constants(; N::Vector{Int}, K::Int, L::Dict{Bool, Int}, J::Int,
                   yQuantiles::Vector{Float64}, pBounds::Vector{Float64},
                   priors::Priors, tau::Float64=.005,
                   use_stickbreak::Bool=false, noisy_var::Float64=10.0)
  I = length(N)
  # beta = ???
end
