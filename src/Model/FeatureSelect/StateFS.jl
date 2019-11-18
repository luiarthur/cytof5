"""
Parameter state in Gibbs sampler for feature allocation model
with feature selection. This augments `State` with parameters:
`r`, `W_star`, and `p`-or-`omega`.

Usage example:
==============

```julia
julia> s = StateFS{Float64}()
julia> s.r = rand(Bool, 3, 10)
```

NOTE:
=====
The update order should be:
Z -> v -> alpha -> p -> r -> lam -> W* -> gamma -> eta -> delta -> sig^2 -> y*

Particularly, `lam` must be updated before W* and preferrably after p & r.
And `gamma` should be updated after Z and before delta.
"""
mutable struct StateFS{F <: AbstractFloat}
  r::Matrix{Bool}  # dim: IxK
  W_star::Matrix{F}  # dim: IxK
  omega::Vector{F}  # dim: number of covariates (`length(x_i)`) + 1
  theta::State{F}  # State

  # Primary constructor returns an uninitialized stateFS.
  StateFS{F}() where {F <: AbstractFloat} = new()
end


function StateFS{F}(theta::State{F}, dfs::DataFS) where {F <: AbstractFloat}
  sfs = StateFS{F}()
  I, K = size(theta.W)

  sfs.theta = theta
  # sfs.r = rand(Bool, I, K)
  sfs.r = ones(Bool, I, K)
  # sfs.W_star = ones(I, K)
  sfs.W_star = theta.W * 5
  sfs.omega = zeros(dfs.P)

  return sfs
end
