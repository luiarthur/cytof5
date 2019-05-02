# Model Constants
struct Constants{F <: AbstractFloat}
  I::Int
  N::Vector{Int}
  J::Int
  K::Int
  L::Dict{Bool, Int}
  tau::F
  use_stickbreak::Bool
end


