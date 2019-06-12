# TODO: NOT COMPLETE! DO NOT USE!!!
#=
function table(x::AbstractVector, y::AbstractVector)
  n = length(x)
  @assert length(y) == n
  
  ux = unique(x)
  uy = unique(y)

  M = Dict{Tuple{Int, Int}, Int}()

  for i in ux
    for j in uy
      M[(i, j)] = sum(y .== j .& x .== i)
    end
  end

  return M
end

"""
Adjusted rand index ∈ [0, 1] between two sets
of cluster labels (x and y)
"""
function ari(x::AbstractVector, y::AbstractVector)
  n = length(x)
  @assert length(y) == n
  ux = unique(x)
  uy = unique(y)
  ariindex = 0
  expindex = 0
  maxindex = 0

  for i in ux
    for j in jy
      nij = 0
    end
  end
end
=#
