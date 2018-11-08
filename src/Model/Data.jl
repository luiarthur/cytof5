#const VecMissMat = Vector{Matrix{Union{T, Missing}}} where T

function nrow(m::Matrix)
  return size(m, 1)
end

function ncol(m::Matrix)
  return size(m, 2)
end

@ann struct Data
  #y::VecMissMat{Float64}
  y::Vector{Matrix{Float64}}
  I::Int
  J::Int
  N::Vector{Int}
  m::Vector{Matrix{Int8}}

  function Data(y) 
    @assert all( ncol.(y) .== ncol(y[1]) )
    I = length(y)
    J = ncol(y[1])
    N = nrow.(y)
    m = [Int8(1) * isnan.(y[i]) for i in 1:I]
    return new(y, I, J, N, m)
  end
end

include("missing_mechanism.jl")
include("generate_data.jl")
