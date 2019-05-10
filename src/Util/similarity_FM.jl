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
- W1: point estimate of W from FAM (with K1 columns)
- Z2: point estimate of Z from FAM (with K2 columns)
"""
function similarity_FM(Z1::Matrix{T}, W1::Matrix{F}, Z2::Matrix{T};
                       z_diff=sumabsdiff) where {T <: Integer, F <: AbstractFloat}
  # cache dimensions
  I, K1 = size(W1)
  J = size(Z1, 1)
  K2 = size(Z2, 2)

  # Make sure dimensions match
  @assert size(Z1, 1) == size(Z2, 1) == J
  @assert size(Z1, 2) == size(W1, 2) == K1

  # Iterate over columns of Z1 -- the larger matrix
  @assert K1 >= K2
   
  function engine(z1::Matrix{T}, w1::Vector{F}, z2::Matrix{T}; s::Float64=0.0)
    if length(z1) == 0
      return s
    elseif length(z2) == 0
      return engine(z1[:, 2:end], w1[2:end], Z2, s=s)
    else
      k2, d = closest_column(z1[:, 1], z2, z_diff=z_diff)
      engine(z1[:, 2:end], w1[2:end], z2[:, 1:end .!= k2], s=s+d*w1[1])
    end
  end

  s = sum(begin 
            ord_1 = sortperm(W1[i, :], rev=true)
            engine(Z1[:, ord_1], W1[i, ord_1], Z2)
          end for i in 1:I)
  
  return s
end
