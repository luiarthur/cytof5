module StickBreak

using Flux
using Flux.Tracker: TrackedArray, @grad, track

# https://discourse.julialang.org/t/how-to-create-tracked-cumsum-using-flux-jl/17772/2

# cumsum
Base.cumsum(x::TrackedArray; dims=1) = track(cumsum, x, dims)
@grad cumsum(x::TrackedArray, dims) = 
    cumsum(x.data, dims=dims), g -> ( reverse(cumsum(reverse(g, dims=dims), dims=dims), dims=dims) , nothing)

# cumprod
Base.cumprod(x::TrackedArray; dims=1) = track(cumprod, x, dims)
@grad cumprod(x::TrackedArray, dims) = 
    cumprod(x.data, dims=dims), g -> ( reverse(cumprod(reverse(g, dims=dims), dims=dims), dims=dims) , nothing)
    
"""
x: real vector of dim K - 1
return: simplex of dim K
"""
function transform(x::T) where T
  K = length(x) + 1
  k_vec = collect(1:K-1)
  z = sigmoid.(x .- log.(K .- k_vec))
  one_minus_z_cumprod = cumprod(1.0 .- z)

  p = vcat(z, 1.0) .* vcat(1.0, one_minus_z_cumprod)
  # return p
  return Tracker.collect(p)
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

