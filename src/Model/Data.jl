const VecMissMat = Vector{Matrix{Union{T, Missing}}} where T

function nrow(m)
  return size(m, 1)
end

function ncol(m)
  return size(m, 2)
end

struct Data
  y::VecMissMat{Float16}
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

