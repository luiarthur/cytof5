"""
DataFS object for FAM with feature selection.
Augments Data object with covariates X.
X can be indicators, or continuous predictors.
"""
@namedargs struct DataFS
  X::Matrix{Float64}  # dim: I x P
  P::Int  # P is number of covariates, aka ncol(X)
  data::Data

  function DataFS(X, data::Data)
    P = ncol(X)
    return new(X, P, data)
  end
end

