const Cube = Array{T, 3} where T

mutable struct State
  Z::Matrix{Int8} # Dim: J x K. Z[j,k] ∈ {0, 1}
  mus::Dict{Int8, Matrix{Float16}}
  alpha::Float16
  v::Vector{Float16}
  W::Matrix{Float16}
  sig2::Vector{Float16}
  eta::Dict{Int8, Cube{Float16}}
  lam::Vector{Vector{Int16}} # Array of Array. lam[1:I] ∈ {1,...,K}
  gam::Vector{Matrix{Float16}}
  y_imputed::Vector{Matrix{Float16}}
  b0::Float16
  b1::Float16

  State(;
        Z::Any = missing,
        mus::Any = missing,
        alpha::Any = missing, 
        v::Any = missing,
        W::Any = missing,
        sig2::Any = missing,
        eta::Any = missing,
        lam::Any = missing,
        gam::Any = missing,
        y_imputed::Any = missing,
        b0::Any = missing,
        b1::Any = missing) = new(Z, mus, alpha, v, W, sig2, eta, lam, gam,
                                 y_imputed, b0, b1)

end

