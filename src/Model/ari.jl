# TODO: Needs testing
# Taken from Clustering.jl/src/randindex.jl)

struct Contingency
  t::Matrix{Int}
  x::Vector{Int}
  y::Vector{Int}
end

function Contingency(M::Dict{Tuple{Int, Int}, Int})
  x, y = zip(keys(M)...)
  ux = sort(unique(x))
  uy = sort(unique(y))
  t = zeros(Int, length(ux), length(uy))
  for i in ux
    for j in uy
      idx_i = findfirst(index -> index == i, ux)
      idx_j = findfirst(index -> index == j, uy)
      if haskey(M, (i, j))
        t[idx_i, idx_j] = M[i, j] + 0
      end
    end
  end
  return Contingency(t, ux, uy)
end


function table(x::AbstractVector, y::AbstractVector)
  n = length(x)
  @assert length(y) == n
  
  range_x = minimum(x):maximum(x)
  range_y = minimum(y):maximum(y)
  C = StatsBase.counts(x, y, (range_x, range_y)) 

  M = Dict{Tuple{Int, Int}, Int}()
  for i in 1:length(range_x)
    for j in 1:length(range_y)
      c_ij = C[i, j]
      if c_ij > 0
        M[(range_x[i], range_y[j])] = C[i, j]
      end
    end
  end

  return M
end

ctable(x, y) = Contingency(table(x, y))

function pretty(C::Contingency)
  M = zeros(Union{Int, Missing}, length(C.x) + 1, length(C.y) + 1)
  M[2:end, 2:end] .= C.t
  M[2:end, 1] = C.x
  M[1, 2:end] = C.y
  M[1, 1] = missing
  M
end

#= TEST
x = [1,3,5,2,4,5,5,1,1]
y = [2,4,6,1,1,1,1,1,2]
C = ctable(x, y)

C.t
C.x
C.y
pretty(C)
=#

"""
Adjusted rand index ∈ [0, 1] between two sets
of cluster labels (x and y)
"""
function ari(x::AbstractVector, y::AbstractVector)
  n = length(x)
  @assert length(y) == n

  C = ctable(x, y)
  c = C.t

  n = round(Int,sum(c))
  nis = sum(abs2, sum(c, dims=2))        # sum of squares of sums of rows
  njs = sum(abs2, sum(c, dims=1))        # sum of squares of sums of columns

  t1 = binomial(n,2)            # total number of pairs of entities
  t2 = sum(c.^2)                # sum over rows & columnns of nij^2
  t3 = .5*(nis+njs)

  # Expected index (for adjustment)
  nc = (n*(n^2+1)-(n+1)*nis-(n+1)*njs+2*(nis*njs)/n)/(2*(n-1))

  A = t1+t2-t3;        # no. agreements
  D = -t2+t3;          # no. disagreements

  if t1 == nc
      # avoid division by zero; if k=1, define Rand = 0
      ARI = 0
  else
      # adjusted Rand - Hubert & Arabie 1985
      ARI = (A-nc)/(t1-nc)
  end

  return ARI
end
