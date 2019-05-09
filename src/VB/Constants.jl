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
function Constants(; y::Vector{Matrix{F}}, K::Int, L::Dict{Bool, Int}, 
                   yQuantiles::Vector{Float64}, pBounds::Vector{Float64},
                   tau::Float64=.005,
                   yBounds::Union{Missing, Vector{Float64}}=missing,
                   use_stickbreak::Bool=false,
                   noisy_var::Float64=10.0) where {F <: AbstractFloat}
  N = size.(y, 1)
  J = size(y[1], 2)
  I = length(N)

  beta = begin
    if ismissing(yBounds)
      b = [gen_beta_est(yi[yi .< 0], yQuantiles, pBounds) for yi in y]
    else
      b = [solveBeta(yBounds, pBounds) for i in 1:I]
    end
    b
  end

  priors = Priors(K=K, L=L, use_stickbreak=use_stickbreak)
  return Constants(I, N, J, K, L, tau, beta, use_stickbreak, noisy_var, priors)
end

function printConstants(c::Constants, preprintln::Bool=true)
  if preprintln
    println()
  end

  println("Constants:")
  for f in fieldnames(typeof(c))
    if f != :priors
      println("$f => $(getfield(c, f))")
    end
  end

  println("Priors:")
  for p in fieldnames(typeof(c.priors))
    println("$p => $(getfield(c.priors, p))")
  end
end
