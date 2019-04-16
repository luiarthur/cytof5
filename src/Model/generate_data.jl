"""
create (n x n) identity matrix

# Arguments
`T`: type of the matrix elements

`n`: dimensions
"""
function eye(T, n::Int)
  return Matrix{T}(LinearAlgebra.I, n, n)
end


"""
left order a binary matrix (Z)

convenience function for:
```julia
julia> sortslices(Z, dims=2, rev=true)
```
"""
function leftOrder(Z::Matrix{T}) where T
  return sortslices(Z, dims=2, rev=true)
end


"""
Generate a simple Z matrix. For simulation
studies. J, K are integers and dimensions
of the (binary) Z matrix. J must be
a multiple of K.
"""
function genSimpleZ(J::Int, K::Int)
  g = div(J, K)
  @assert g == J / K

  return kron(eye(Int, K), fill(1,g))
end

"""
Assert that a Z matrix is valid
"""
function isValidZ(Z)
  hasRepeatedColumns = size(unique(Z, dims=2), 2) < size(Z, 2)
  hasColOfZero =  any(sum(Z, dims=1) .== 0)
  return  !(hasRepeatedColumns || hasRepeatedColumns)
end

"""
Generate a (J x K) Z matrix for simulation studies.

`prob1` is the desired proportion of 1's in the matrix.
"""
function genZ(J::Int, K::Int, prob1::Float64)::Matrix{Int}
  @assert 0 < prob1 < 1
  Z = Int.(rand(J, K) .> prob1)
  Z = sortslices(Z, dims=1, rev=true)
  Z = leftOrder(Z)

  # if size(unique(Z, dims=2), 2) < K || any(sum(Z, dims=1) .== 0) || all(sum(Z, dims=2) .== 0)
  if isValidZ(Z)
    return Z
  else
    return genZ(J, K, prob1)
  end
end

#= Test
Z = Int.(randn(3,5) .> 0)
=#

function genData(J::Int, N::Vector{Int}, K::Int, L::Dict{Int, Int};
                 useSimpleZ::Bool=true, prob1::Float64=.6,
                 sortLambda::Bool=false, propMissingScale::Float64=0.7)

  I = length(N)
  Z = useSimpleZ ? genSimpleZ(J, K) : genZ(J, K, prob1)

  genData(J=J, N=N, K=K, L=L, Z=Z,
          beta=[-9.2, -2.3], # linear missing mechanism
          sig2=fill(0.1, I),
          mus=Dict(0=>collect(range(-5, length=L[0], stop=-1)),
                   1=>collect(range(1, length=L[1], stop=5))),
          a_W=[float(i) for i in 1:K],
          a_eta=Dict([ z => [float(l) for l in 1:L[z]] for z in 0:1 ]),
          sortLambda=sortLambda, propMissingScale=propMissingScale)
end

function genData(; J::Int, N::Vector{Int}, K::Int, L::Dict{Int, Int},
                 Z::Matrix{Int}, beta::Vector{Float64},
                 sig2::Vector{Float64}, mus::Dict{Int, Vector{Float64}}, 
                 a_W::Vector{Float64}, a_eta::Dict{Int, Vector{Float64}},
                 sortLambda::Bool=false, propMissingScale::Float64=0.7,
                 eps=zeros(length(N)))

  # Check eps
  @assert length(eps) == length(N)
  @assert all(0 .<= eps .<= 1)

  # Check Z dimensions
  @assert ncol(Z) == K && nrow(Z) == J

  # Check N dimensions
  @assert all(N .> 0)

  I = length(N)

  # Check sig2 dimensions
  @assert all(sig2 .> 0)
  @assert length(sig2) == I

  # Check mus dimensions
  @assert all(mus[0] .< 0) && all(mus[1] .> 0)
  @assert length(mus[0]) == L[0] && length(mus[1]) == L[1]

  # Check a_W dimensions
  @assert length(a_W) == K
  @assert all(a_W .> 0)

  # Check a_eta dimensions
  @assert length(a_eta[0]) == L[0]
  @assert length(a_eta[1]) == L[1]
  @assert all(a_eta[0] .> 0)
  @assert all(a_eta[1] .> 0)

  # Simulate W
  W = zeros(I, K)
  for i in 1:I
    W[i,:] = rand(Dirichlet(Random.shuffle(a_W)))
  end


  # Simulate lambda
  lam = [begin
           p = [eps[i]; (1 - eps[i]) * W[i, :]]
           rand(Categorical(p), N[i]) .- 1
         end for i in 1:I]
  if sortLambda
    lam = [sort(lami) for lami in lam]
  end

  # Simulate eta
  eta = Dict([ z => zeros(Float64, I, J, L[z]) for z in 0:1 ])
  for i in 1:I
    for j in 1:J
      eta[0][i, j, :] = rand(Dirichlet(Random.shuffle(a_eta[0])))
      eta[1][i, j, :] = rand(Dirichlet(Random.shuffle(a_eta[1])))
    end
  end

  # Simulate gam
  gam = [zeros(Int, Ni, J) for Ni in N]
  for i in 1:I
    for j in 1:J
      for n in 1:N[i]
        k = lam[i][n]
        if k > 0
          z = Z[j, k]
          gam[i][n,j] = rand(Categorical(eta[z][i, j, :]))
        else
          gam[i][n,j] = 0
        end
      end
    end
  end

  # Generate y
  #y = [zeros(Union{Float64, Missing}, Ni, J) for Ni in N]
  y = [zeros(Float64, Ni, J) for Ni in N]
  y_complete = [zeros(Float64, Ni, J) for Ni in N]

  function z_get(i, n, j)
    return Z[j, lam[i][n]]
  end

  function mu_get(i, n, j)
    l = gam[i][n, j]
    if l > 0
      z = z_get(i, n, j)
      out = mus[z][l] 
    else
      out = 0.0
    end
    return out
  end

  for i in 1:I
    for j in 1:J
      for n in 1:N[i]
        sig_i = lam[i][n] > 0 ? sqrt(sig2[i]) : 3.0
        y_complete[i][n, j] = rand(Normal(mu_get(i, n, j), sig_i))
      end

      # Set some to be missing
      p_miss = [prob_miss(y_complete[i][n, j], beta) for n in 1:N[i]]
      prop_missing = rand() * propMissingScale * sum(W[i,:] .* (1 .- Z[j,:]))
      num_missing = Int(round(N[i] * prop_missing))
      idx_missing = Distributions.wsample(1:N[i], p_miss, num_missing, replace=false)
      y[i][:, j] .= y_complete[i][:, j] .+ 0
      y[i][idx_missing, j] .= NaN
    end
  end

  return Dict(:y=>y, :y_complete=>y_complete, :Z=>Z, :W=>W,
              :eta=>eta, :mus=>mus, :sig2=>sig2, :lam=>lam, :gam=>gam,
              :beta=>beta, :eps=>eps)
end # genData

#precompile(genData, (String, Int, Vector{Int}, Int, Int, Bool, Float64, Bool, Float64))

