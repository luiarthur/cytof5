using Flux
using Flux.Tracker: TrackedArray, @grad, track

# NOTE:
# See "Vector arguments and functions" and "Automatic differentiation using
# dual numbers" in this page and do the derivations for
# `cumsum` and `cumprod` by hand.
# https://en.wikipedia.org/wiki/Automatic_differentiation

# Also, for x (input of dim M) and y (input of dim N). During backward, Δ
# should have dimensions (N) of output. We compute J (N × M) where
# Jₙₘ = ∂yₘ/∂xₙ. Finally, we return J • Δ. See also `jacobian` in:
# https://github.com/FluxML/Tracker.jl/blob/master/src/back.jl

# cumsum
# https://discourse.julialang.org/t/how-to-create-tracked-cumsum-using-flux-jl/17772/2
Base.cumsum(x::TrackedArray; dims=1) = track(cumsum, x, dims)
@grad function cumsum(x::TrackedArray, dims)
  return cumsum(x.data, dims=dims), function(Δ)
    return (reverse(cumsum(reverse(Δ, dims=dims), dims=dims), dims=dims), nothing)
  end
end

# cumprod
# https://stackoverflow.com/questions/40916955/how-to-compute-gradient-of-cumprod-safely
Base.cumprod(x::TrackedArray; dims=1) = track(cumprod, x, dims)
@grad function cumprod(x::TrackedArray, dims)
  return cumprod(x.data, dims=dims), function(Δ)
    # reverse(cumsum(reverse(cumprod_x .* Δ, dims=dims), dims=dims), dims=dims) ./ x.data, nothing
    reverse(cumsum(reverse(cumprod(x, dims=dims) .* Δ, dims=dims), dims=dims), dims=dims) ./ x, nothing
  end
end

# reverse
# https://github.com/FluxML/Tracker.jl/blob/master/src/lib/array.jl
Base.reverse(x::TrackedArray; dims=1) = track(reverse, x, dims)
@grad function reverse(x::TrackedArray, dims)
  return reverse(x.data, dims=dims), function(Δ)
    println("in reverse backprop")
    return reverse(Δ, dims=dims), nothing
  end
end

function head(x::T) where {T <: AbstractArray}
  """
  does the same thing as `x[..., :-1]` in pytorch.
  """
  return x[[axes(x, i) for i in 1:ndims(x)-1]..., 1:end-1]
end

function layer(x::T, l::Integer) where {T <: AbstractArray}
  """
  returns: x[[axes(x, i) for i in 1:ndims(x)-1]..., l]
  """
  return x[[axes(x, i) for i in 1:ndims(x)-1]..., l]
end

# deprecated:
# cumprod_pos(x::T; dims=1) where T = exp.(cumsum(log.(x), dims=dims))
 
# Also see this:
# https://github.com/FluxML/Flux.jl/pull/524
