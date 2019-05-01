"""
x: real vector of dim K - 1
return: simplex of dim K
"""
function SB_transform(x::T) where T
  size_x = size(x)
  ndim = ndims(x)
  dim_head = size_x[1:end-1]
  K = size_x[end] + 1
  ks = cumsum(one.(x), dims=ndim)
  z = sigmoid.(x .- log.(K .- ks))
  one_minus_z_cumprod = cumprod(1 .- z, dims=ndim)

  if ndim == 1
    ones_pad =  Tracker.collect([one.(slice(x, 1))])
    p = vcat(z, ones_pad) .* vcat(ones_pad, one_minus_z_cumprod)
  else
    ones_pad = one.(slice(x, 1))
    p = cat(z, ones_pad, dims=ndim) .* cat(ones_pad, one_minus_z_cumprod, dims=ndim)
  end

  return p
end


"""
x: real vector of dim K - 1
p: simplex of dim K
return: log abs value of determinant of x
"""
function SB_logabsdetJ(x, p)
  ndim = ndims(x)
  K = size(p)[end]
  ks = cumsum(one.(x), dims=ndim)
  z = sigmoid.(x .- log.(K .- ks))
  detJ = sum(log1p.(-z) .+ log.(head(p)), dims=ndim)

  return detJ
end
