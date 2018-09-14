const VecMissMat = Vector{Matrix{Union{T, Missing}}} where T

function nrow(m::Matrix)
  return size(m, 1)
end

function ncol(m::Matrix)
  return size(m, 2)
end

struct Data
  y::VecMissMat{Float64}
  I::Int
  J::Int
  N::Vector{Int}

  function Data(y) 
    @assert all( ncol.(y) .== ncol(y[1]) )
    I = length(y)
    J = ncol(y[1])
    N = nrow.(y)
    return new(y, I, J, N)
  end
end

include("missing_mechanism.jl")
include("generate_data.jl")
