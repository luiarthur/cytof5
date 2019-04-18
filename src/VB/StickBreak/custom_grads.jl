using Flux
using Flux.Tracker: TrackedArray, @grad, track

# Also see this:
# https://github.com/FluxML/Flux.jl/pull/524

# TODO:
# remove this file when Zygote.jl is officially the AD for Flux

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
    reverse(cumsum(reverse(cumprod(x, dims=dims) .* Δ, dims=dims), dims=dims), dims=dims) ./ x, nothing
  end
end

# reverse
# https://github.com/FluxML/Tracker.jl/blob/master/src/lib/array.jl
Base.reverse(x::TrackedArray; dims=1) = track(reverse, x, dims)
@grad function reverse(x::TrackedArray, dims)
  return reverse(x.data, dims=dims), function(Δ)
    return reverse(Δ, dims=dims), nothing
  end
end

function head(x::T) where {T <: AbstractArray}
  """
  does the same thing as `x[..., :-1]` in pytorch.
  """
  return x[[axes(x, i) for i in 1:ndims(x)-1]..., 1:end-1]
end

function slice(x::T, l::Integer) where {T <: AbstractArray}
  """
  returns: x[[axes(x, i) for i in 1:ndims(x)-1]..., l]
  """
  return x[[axes(x, i) for i in 1:ndims(x)-1]..., l]
end


function logsumexp(logx::T; dims::Integer=1) where T
  """
  logsumexp
  """
  mx = maximum(logx, dims=dims)
  return mx .+ log.(sum(exp.(logx .- mx), dims=dims))
end


