using Flux
using Flux.Tracker: TrackedArray, @grad, track

# cumsum
# https://discourse.julialang.org/t/how-to-create-tracked-cumsum-using-flux-jl/17772/2
Base.cumsum(x::TrackedArray; dims=1) = track(cumsum, x, dims)
@grad function cumsum(x::TrackedArray, dims)
  return cumsum(x.data, dims=dims), function(Δ)
    return (reverse(cumsum(reverse(Δ, dims=dims), dims=dims), dims=dims), nothing)
  end
end

# cumprod positive
# https://stackoverflow.com/questions/40916955/how-to-compute-gradient-of-cumprod-safely
Base.cumprod(x::TrackedArray; dims=1) = track(cumprod, x, dims)
@grad function cumprod(x::TrackedArray, dims)
  cumprod_x = cumprod(x.data, dims=dims)
  return cumprod_x, function(Δ)
    return reverse(cumsum(reverse(cumprod_x .* Δ, dims=dims), dims=dims), dims=dims) ./ x.data, nothing
  end
end

# deprecated:
# cumprod_pos(x::T; dims=1) where T = exp.(cumsum(log.(x), dims=dims))
 
