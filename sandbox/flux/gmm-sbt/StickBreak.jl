module StickBreak

using Flux
using Flux.Tracker: TrackedArray, @grad, track

# https://discourse.julialang.org/t/how-to-create-tracked-cumsum-using-flux-jl/17772/2

# cumsum
Base.cumsum(x::TrackedArray; dims=1) = track(cumsum, x, dims)
@grad function cumsum(x::TrackedArray, dims)
  return cumsum(x.data, dims=dims), function(Δ)
    return (reverse(cumsum(reverse(Δ, dims=dims), dims=dims), dims=dims), nothing)
  end
end

# cumprod positivce
cumprod_pos(x::T; dims=1) where T = exp.(cumsum(log.(x), dims=dims))
    
"""
x: real vector of dim K - 1
return: simplex of dim K
"""
function transform(x::T) where T
  K = length(x) + 1
  k_vec = collect(1:K-1)
  z = sigmoid.(x .- log.(K .- k_vec))
  one_minus_z_cumprod = cumprod_pos(1.0 .- z)

  # VERSION I
  p = vcat(z, [1.0]) .* vcat([1.0], one_minus_z_cumprod)
  return p

  # VERSION II
  # p = z .* vcat(1.0, one_minus_z_cumprod[1:end-1])
  # return vcat(p, 1.0 - sum(p))
  # return Tracker.collect(vcat(p, 1.0 - sum(p)))

  # VERSION III
  # p = [begin
  #        if k > 1
  #          z[k] * one_minus_z_cumprod[k - 1]
  #        else
  #          z[1]
  #        end
  #      end for k in 1:K-1]
  # return Tracker.collect(vcat(p, 1.0 - sum(p)))
end


"""
x: real vector of dim K - 1
p: simplex of dim K
return: log abs value of determinant of x
"""
function logabsdetJ(x, p)
  K = length(p)
  k_vec = collect(1:K-1)
  z = sigmoid.(x .- log.(K .- k_vec))
  detJ = sum(log1p.(-z) .+ log.(p[1:end-1]))

  return detJ
end

end # StickBreak

