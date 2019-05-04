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

