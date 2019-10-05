"""
DataFS object for FAM with feature selection.
Augments Data object with covariates X.
X can be indicators, or continuous predictors.
"""
struct DataFS
  X::Matrix{Float64}  # dim: I x P
  P::Int  # P is number of covariates, aka ncol(X)
  data::Data
end

function DataFS(data::Data, X::Matrix{Float64})
  P = ncol(X)
  return DataFS(X, P, data)
end
