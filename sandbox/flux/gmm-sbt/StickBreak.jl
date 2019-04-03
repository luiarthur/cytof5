module StickBreak
using Flux
using Flux.Tracker: TrackedArray, @grad, track

include("custom_grads.jl") # cumsum, cumprod
   
"""
x: real vector of dim K - 1
return: simplex of dim K
"""
function transform(x::T) where T
  K = length(x) + 1
  k_vec = collect(1:K-1)
  z = sigmoid.(x .- log.(K .- k_vec))
  one_minus_z_cumprod = cumprod(1.0 .- z)
  p = vcat(z, [1.0]) .* vcat([1.0], one_minus_z_cumprod)

  return p
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

