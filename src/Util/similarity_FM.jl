sumabsdiff(z1::Vector{T}, z2::Vector{T}) where T <: Integer = sum(abs.(z1 - z2))

function meanabsdiff(z1::Vector{T}, z2::Vector{T}) where T <: Integer
  return sumabsdiff(z1, z2) / size(z1, 1)
end

function closest_column(z1::Vector{T}, Z2::Matrix{T};
                        z_diff=sumabsdiff) where T<: Integer

  # Compute distance between each column
  d = [z_diff(z1, Z2[:, k2]) for k2 in 1:size(Z2, 2)]

  k2 = argmin(d)
  return k2, d[k2]
end

"""
compute similarity of two feature matrices

Arguments
=========
- Z1: point estimate of Z from FAM (with K1 columns)
- Z2: point estimate of Z from FAM (with K2 columns)
- W1: point estimate of W from FAM (with K1 columns)
- W2: point estimate of Z from FAM (with K2 columns)
"""
function similarity_FM(Z1::Matrix{T}, Z2::Matrix{T},
                       W1::Matrix{F}, W2::Matrix{F};
                       z_diff=sumabsdiff) where {T <: Integer, F <: AbstractFloat}
  # cache dimensions
  I, K1 = size(W1)
  J = size(Z1, 1)
  K2 = size(W2, 2)

  # Make sure dimensions match
  @assert size(Z1, 1) == size(Z2, 1) == J
  @assert size(Z1, 2) == size(W1, 2) == K1
  @assert size(Z2, 2) == size(W2, 2) == K2
  @assert size(W1, 1) == size(W2, 1) == I

  # Iterate over columns of Z1 -- the smaller matrix
  @assert K1 <= K2
   
  function engine(Z1::Matrix{T}, Z2::Matrix{T},
                  W1::Vector{F}, W2::Vector{F}; s::Float64=0.0)
    if length(Z1) == 0
      return s
    else
      k2, d = closest_column(Z1[:, 1], Z2, z_diff=z_diff)
      engine(Z1[:, 2:end], Z2[:, 1:end .!= k2],
             W1[2:end], W2[1:end .!= k2], s=s+d*W1[1])
    end
  end

  s = sum(engine(Z1, Z2, W1[i, :], W2[i, :]) for i in 1:I)
  
  return s
end
