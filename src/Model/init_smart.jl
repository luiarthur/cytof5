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
    return y
  end
end

function preimpute!(y::Matrix{T}, missMean::AbstractFloat=6.0) where {T <: Number}
  num_missing = sum(isnan.(y))
  y[isnan.(y)] .= randn(num_missing) .- missMean
end


function precluster(y::Matrix{T}; K::Int, modelNames::String="VVI") where {T <: Number}
  return Mclust(y, G=K, modelNames=modelNames)
end

#= Test
using JLD2, FileIO
function loadSingleObj(objPath)
  data = load(objPath)
  return data[collect(keys(data))[1]]
end
include("../../sims/sim_study/util.jl")

y = loadSingleObj("../../sims/cb/data/cytof_cb_with_nan.jld2")
I = length(y)
preimpute!.(y)
y = subsampleData.(y, .1)
size.(y)

clus = [precluster(y[i], K=5) for i in 1:I]
ord = [sortperm(Int.(clus[i][:classification])) for i in 1:I]

util.myImage(y[1][ord[1], :], util.blueToRed(9), zlim=[-4,4], addL=true, na="black")
util.myImage(y[2][ord[2], :], util.blueToRed(9), zlim=[-4,4], addL=true, na="black")
util.myImage(y[3][ord[3], :], util.blueToRed(9), zlim=[-4,4], addL=true, na="black")

=#
