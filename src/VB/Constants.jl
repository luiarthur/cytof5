# Model Constants
struct Constants{F <: AbstractFloat}
  I::Int
  N::Vector{Int}
  J::Int
  K::Int
  L::Dict{Bool, Int}
  tau::F
  beta::Vector{Vector{F}} # P x I
  use_stickbreak::Bool
  noisy_var::F
  priors::Priors
end

