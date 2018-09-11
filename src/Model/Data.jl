const VecMissMat = Vector{Matrix{Union{T, Missing}}} where T

function nrow(m)
  return size(m, 1)
end

function ncol(m)
  return size(m, 2)
end

struct Data
  y::VecMissMat{Float16}

  function Data(y) 
    @assert all( ncol.(y) .== ncol(y[1]) )
    return new(y)
  end
end

function getI(data::Data)
  return length(data.y)
end

function getJ(data::Data)
  return ncol(data.y[1])
end

function getN(data::Data)
  return nrow.(data.y)
end
