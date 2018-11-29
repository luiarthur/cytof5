# TODO: 
# - test
# - move `using RCall` to Model.jl
# - include this file from Model.jl
# - add RCall, StatsBase to Manifest
module SmartInit

using RCall
using JLD2, FileIO, Distributions
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

function loadSingleObj(objPath)
  data = load(objPath)
  return data[collect(keys(data))[1]]
end

function gen_idx(N)
  I = length(N)
  upper_idx = cumsum(N)
  lower_idx = [1; upper_idx[1:end-1] .+ 1]
  return [lower_idx[i]:upper_idx[i] for i in 1:I]
end


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


function smartInit(y_orig; K::Int, modelNames::String="VVI",
                   missMean=6.0, warn::Bool=true,
                   cluster_samples_jointly::Bool=true) where {T <: Number}

  y = deepcopy(y_orig)
  I = length(y)
  N = size.(y, 1)
  idx = gen_idx(N)

  if cluster_samples_jointly
    y = vcat(y...)
    preimpute!(y, missMean)
    clus = Mclust(y, G=K, modelNames=modelNames, warn=warn)
  else
    preimpute!.(y, missMean)
    clus = [Mclust(y[i], G=K, modelNames=modelNames, warn=warn) for i in 1:I]
  end


  # Get order of class labels
  if cluster_samples_jointly
    lam = [Int.(clus[:classification])[idx[i]] for i in 1:I]
  else
    lam = [Int.(clus[i][:classification]) for i in 1:I]
  end
  ord = [sortperm(lam[i]) for i in 1:I]

  # Get Z
  if cluster_samples_jointly
    # clus_means = [mean(y[idx[i], :][lam[i] .== k, :], dims=1) for k in 1:K, i in 1:I]
    clus_means = [mean(y[vcat(lam...) .== k, :], dims=1) for k in 1:K]
  else
    # clus_means = [mean(y[i][lam[i] .== k, :], dims=1) for k in 1:K, i in 1:I]
    group_sizes = [sum(vcat(lam...) .== k) for k in 1:K]
    clus_sums = [sum(y[i][lam[i] .== k, :], dims=1) for k in 1:K, i in 1:I]
    # FIXME!
    clus_means = [sum(clus_sums[k, :], dims=2) / group_sizes[k] for k in 1:K]
  end
  # Z = [Int8(1) * (Matrix(vcat(clus_means[:, i]...)') .> 0) for i in 1:I]
  Z = Int8(1) * (Matrix(vcat(clus_means...)') .> 0)

  # Unqiue Z
  # println.(size.(unique.(Z, dims=2)))
  # unique(hcat(Z...), dims=2)
  unique(Z, dims=2)

  # Get W
  W = [mean(lam[i] .== k) for i in 1:I, k in 1:K]

  return Dict(:N => N, :lam => lam, :Z => Z, :W => W, :idx => idx, :y => y,
              :cluster_samples_jointly => cluster_samples_jointly)
end

end # module

#= Test 2: Cluster all samples, then separate
include("../../sims/sim_study/util.jl")
y = SmartInit.loadSingleObj("../../sims/cb/data/cytof_cb_with_nan.jld2")
y = SmartInit.subsampleData.(y, .1)

# Cluster things ##########
init = SmartInit.smartInit(y, K=5, cluster_samples_jointly=false)

# Plot yZ
W = init[:W]
Z = init[:Z]
lam = init[:lam]
idx = init[:idx]

util.yZ(y[1], Z, W[1,:], lam[1], using_zero_index=false, thresh=.9, na="black");
util.yZ(y[2], Z, W[2,:], lam[2], using_zero_index=false, thresh=.9, na="black");
util.yZ(y[3], Z, W[3,:], lam[3], using_zero_index=false, thresh=.9, na="black");

util.hist(y[1][:, 7], xlab="", ylab="", main="")
mean(isnan.(y[1][:, 7]))

# Cluster things #########
init = SmartInit.smartInit(y, K=10)

# Plot yZ
W = init[:W]
Z = init[:Z]
lam = init[:lam]
idx = init[:idx]

util.yZ(y[1], Z, W[1,:], lam[1], using_zero_index=false, thresh=.9, na="black");
util.yZ(y[2], Z, W[2,:], lam[2], using_zero_index=false, thresh=.9, na="black");
util.yZ(y[3], Z, W[3,:], lam[3], using_zero_index=false, thresh=.9, na="black");

=#

