# TODO: 
# - test
# - move `using RCall` to Model.jl
# - include this file from Model.jl

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

function preimpute!(y::Matrix{T}) where {T <: Number}
  num_missing = sum(isnan.(y))
  y[isnan.(y)] .= randn(num_missing) .- 6
end

preimpute(y)

function precluster(y, K, modelNames="VVI")
  Mclust(y, G=K, modelNames=modelNames)
end

