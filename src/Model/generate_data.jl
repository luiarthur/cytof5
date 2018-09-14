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
Generate a (J x K) Z matrix for simulation studies.

`prob1` is the desired proportion of 1's in the matrix.
"""
function genZ(J::Int, K::Int, prob1::Float64)
  @assert 0 < prob1 < 1
  Z = Int.(rand(J, K) .> prob1)
  Z = leftOrder(Z)
  Z = sortslices(Z, dims=1, rev=true)

  if size(unique(Z, dims=1), 1) < J || any(sum(Z, dims=1) .== 0) || all(sum(Z, dims=2) .== 0)
    return genZ(J, K, prob1)
  else
    return Z
  end
end

#= Test
Z = Int.(randn(3,5) .> 0)
=#

# TODO: Test
function genData(I::Int, J::Int, N::Vector{Int}, K::Int, L::Int;
                sortLambda::Bool=false, propMissingScale::Float64=0.7)
  genData(I, J, N, K, L, genSimpleZ(J, K), Dict("b0"=>-9.2, "b1"=>-2.3),
          sig2=fill(0.1, I),
          mus=Dict(0=>range(-5, length=L, stop=-1), 
                   1=>range(1, length=L, stop=5)),
          a_W=[float(i) for i in 1:K], a_eta=[float(l) for l in 1:L],
          sortLambda, propMissingScale)
end

function genData(I::Int, J::Int, N::Vector{Int}, K::Int, L::Int,
                 Z::Matrix{Int}, missMechParams::Dict{String,Float64},
                 sig2::Vector, mus::Dict{Int, Vector}, 
                 a_W::Vector, a_eta::Dict{Int, Vector}; sortLambda::Bool,
                 propMissingScale)

  # Check Z dimensions
  @assert ncol(Z) == K && nrow(Z) == J

  # Check N dimensions
  @assert all(N .> 0)
  @assert length(N) == I

  # Check sig2 dimensions
  @assert all(sig2 .> 0)
  @assert length(sig2) == L

  # Check mus dimensions
  @assert all(mus[0] .< 0) && all(mus[1] .> 0)
  @assert length(mus[0]) == L && length(mus[1]) == L

  # Check a_W dimensions
  @assert length(a_W) == K
  @assert all(a_W .> 0)

  # Check a_eta dimensions
  @assert length(a_eta[0]) == L
  @assert length(a_eta[1]) == L
  @assert all(a_eta[0] .> 0)
  @assert all(a_eta[1] .> 0)

  # Simulate W
  W = zeros(I, K)
  for i in 1:I
    W[i,:] = rand(Dirichlet(a_W))
  end

  # Simulate lambda
  lam = [rand(Categorical(W[i,:]), N[i]) for i in 1:I]
  if sort_lambda
    lam = [sort(lami) for lami in lam]
  end

  # Simulate eta
  eta = Dict{Int, Cube{Float64}}
  for i in 1:I
    for j in 1:J
      eta[0][i, j, :] = rand(Dirichlet(Random.shuffle(a_eta[0])))
      eta[1][i, j, :] = rand(Dirichlet(Random.shuffle(a_eta[1])))
    end
  end

  # Simulate gam
  gam = [zeros(Float64, Ni, J) for Ni in N]
  for i in 1:I
    for j in 1:J
      for n in 1:N[i]
        lin = lam[i][n]
        z = Z[j, lin]
        gam[i][n,j] = rand(Categorical(eta[z][i, j, :]))
      end
    end
  end

  # Generate y
  y = [zeros(Union{Float64, Missing}, Ni, J) for Ni in N]
  y_complete = [zeros(Float64, Ni, J) for Ni in N]

  function z_get(i, n, j)
    return Z[j, lam[i][n]]
  end

  function mu_get(i, n, j)
    l = gam[i][n, j]
    z = z_get(i, n, j)
    return mus[z][l] 
  end


  mmp = missMechParams
  for i in 1:I
    for j in 1:J
      for n in 1:N[i]
        y_complete[i][n, j] = rand(Normal(mu_get(i, n, j)))
      end

      # Set some to be missing
      p_miss = prob_miss.(y_complete[i][:,j], mmp["b0"], mmp["b1"])
      prop_missing = rand() * propMissingScale * sum(W[i,:] .* (1 - Z[j,:]))
      num_missing = Int(N[i] * prop_missing)
      idx_missing = [ Distributions.wsample(p_miss) for ii in 1:num_missing]
      y[i][:, j] .= y_complete[i][:, j] .+ 0
      y[i][idx_missing, j] .= missing
    end
  end

  return Dict(:y=>y, :y_complete=>y_complete, :Z=>Z, :W=>W,
              :eta=>eta, :mus=>mus, :sig2=>sig2, :lam=>lam, :gam=>gam,
              :b0=>b0, :b1=>b1)
end # genData

