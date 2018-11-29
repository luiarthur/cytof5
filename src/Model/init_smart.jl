# TODO: 
# - test
# - move `using RCall` to Model.jl
# - include this file from Model.jl
# - add RCall, StatsBase to Manifest

using RCall
import StatsBase

# Install mclust if necessary
R"""
if ("mclust" %in% installed.packages()) {
  library(mclust)
} else {
  cat("Package `mclust` not found. Installing `mclust`...\n")
  install.packages("mclust")
}
"""

# Convenience link
Mclust = R"Mclust"

function subsampleData(y::Matrix{T}, percentage) where {T <: Number}
  if 0 < percentage < 1
    N = size(y, 1)
    N_subsample = round(Int, N * percentage)
    idx_subsample = StatsBase.sample(1:N, N_subsample, replace=true)
    return y[idx_subsample, :]
  else
    println("Info: percentage ∉  (0, 1). Not subsampling.")
    return y
  end
end

function preimpute!(y::Matrix{T}, missMean::AbstractFloat=6.0) where {T <: Number}
  num_missing = sum(isnan.(y))
  y[isnan.(y)] .= randn(num_missing) .- missMean
end


function precluster(y::Matrix{T}; K::Int, modelNames::String="VVI", 
                    warn::Bool=true) where {T <: Number}
  return Mclust(y, G=K, modelNames=modelNames, warn=warn)
end

#= Test 1: One sample
using JLD2, FileIO, Distributions
function loadSingleObj(objPath)
  data = load(objPath)
  return data[collect(keys(data))[1]]
end
include("../../sims/sim_study/util.jl")

y = loadSingleObj("../../sims/cb/data/cytof_cb_with_nan.jld2")
y = subsampleData.(y, 1)
I = length(y)
preimpute!.(y)

size.(y)

# Cluster things
K = 10
clus = [precluster(y[i], K=K) for i in 1:I]

# Get order of class labels
lam = [Int.(clus[i][:classification]) for i in 1:I]
ord = [sortperm(lam[i]) for i in 1:I]

# Get Z
clus_means = [mean(y[i][lam[i] .== k, :], dims=1) for k in 1:K, i in 1:I]
Z = [Int8(1) * (Matrix(vcat(clus_means[:, i]...)') .> 0) for i in 1:I]

# Unqiue Z
println.(size.(unique.(Z, dims=2)))
unique(hcat(Z...), dims=2)

# Get W
W = [mean(lam[i] .== k) for i in 1:I, k in 1:K]


# Plot yZ
util.yZ(y[1], Z[1], W[1, :], lam[1], using_zero_index=false, thresh=.9);
util.yZ(y[2], Z[2], W[2, :], lam[2], using_zero_index=false, thresh=.9);
util.yZ(y[3], Z[3], W[3, :], lam[3], using_zero_index=false, thresh=.9);
=#

#= Test 2: separate samples
using JLD2, FileIO, Distributions
function loadSingleObj(objPath)
  data = load(objPath)
  return data[collect(keys(data))[1]]
end
include("../../sims/sim_study/util.jl")

function gen_idx(N)
  I = length(N)
  upper_idx = cumsum(N)
  lower_idx = [1; upper_idx[1:end-1] .+ 1]
  return [lower_idx[i]:upper_idx[i] for i in 1:I]
end

y = loadSingleObj("../../sims/cb/data/cytof_cb_with_nan.jld2")
y = subsampleData.(y, .1)
I = length(y)
N = size.(y, 1)
idx = gen_idx(N)
y = vcat(y...)
preimpute!(y)

# Cluster things
K = 10
clus = precluster(y, K=K)

# Get order of class labels
lam = [Int.(clus[:classification])[idx[i]] for i in 1:I]
ord = [sortperm(lam[i]) for i in 1:I]

# Get Z
clus_means = [mean(y[idx[i], :][lam[i] .== k, :], dims=1) for k in 1:K, i in 1:I]
Z = [Int8(1) * (Matrix(vcat(clus_means[:, i]...)') .> 0) for i in 1:I]

# Unqiue Z
println.(size.(unique.(Z, dims=2)))
unique(hcat(Z...), dims=2)

# Get W
W = [mean(lam[i] .== k) for i in 1:I, k in 1:K]


# Plot yZ
util.yZ(y[idx[1], :], Z[1], W[1, :], lam[1], using_zero_index=false, thresh=.9);
util.yZ(y[idx[2], :], Z[2], W[2, :], lam[2], using_zero_index=false, thresh=.9);
util.yZ(y[idx[3], :], Z[3], W[3, :], lam[3], using_zero_index=false, thresh=.9);
=
