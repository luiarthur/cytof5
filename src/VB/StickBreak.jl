module StickBreak
include("custom_grads.jl") # cumsum, cumprod, reverse, head
   
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

  # FIXME: use `one` to convert to type stable function
  p = cat(z, ones(dim_head), dims=ndim) .* cat(ones(dim_head), one_minus_z_cumprod, dims=ndim)

  return p
end


"""
x: real vector of dim K - 1
p: simplex of dim K
return: log abs value of determinant of x
"""
function logabsdetJ(x, p)
  # FIXME: use `one` to convert to type stable function
  ndim = ndims(x)
  K = size(p)[end]
  ks = cumsum(one.(x), dims=ndim)
  z = sigmoid.(x .- log.(K .- ks))
  detJ = sum(log1p.(-z) .+ log.(head(p)), dims=ndim)

  return detJ
end

end # StickBreak

