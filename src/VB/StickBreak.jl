module StickBreak
include("custom_grads.jl") # cumsum, cumprod
   
"""
x: real vector of dim K - 1
return: simplex of dim K
"""
function transform(x::T) where T
  size_x = size(x)
  ndim = ndims(x)
  dim_head = size_x[1:end-1]
  K = size_x[end] + 1
  ks = cumsum(one.(x), dims=ndim)
  z = sigmoid.(x .- log.(K .- ks))
  one_minus_z_cumprod = cumprod(1 .- z, dims=ndim)
  ones_pad = one.(layer(x, 1))
  println(ones_pad)
  p = cat(z, ones_pad, dims=ndim) .* cat(ones_pad, one_minus_z_cumprod, dims=ndim)

  return p
end


"""
x: real vector of dim K - 1
p: simplex of dim K
return: log abs value of determinant of x
"""
function logabsdetJ(x, p)
  ndim = ndims(x)
  K = size(p)[end]
  ks = cumsum(one.(x), dims=ndim)
  z = sigmoid.(x .- log.(K .- ks))
  detJ = sum(log1p.(-z) .+ log.(head(p)), dims=ndim)

  return detJ
end

end # StickBreak

