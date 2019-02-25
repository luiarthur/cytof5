function sbt(x)
  size_x = size(x) 
  dims = length(size_x)
  K = size_x[dims]

  offset = (K + 1) .- cumsum(ones(size_x), dims=dims)
  z = sigmoid.(x .- log.(offset)) 
  z_cumprod = cumprod(1 .- z, dims=dims)
  pad = ones(size_x[1:end-1])
  y = cat(z, pad, dims=dims) .* cat(pad, z_cumprod, dims=dims)

  return y
end

function sbt_inv(p)
  # see: https://pytorch.org/docs/stable/_modules/torch/distributions/transforms.html
  error("unimplemented")
end

function sbt_logabsdetJ(x, p)
  size_x = size(x) 
  dims = length(size_x)
  K = size_x[dims]

  offset = (K + 1) .- cumsum(ones(size_x), dims=dims)
  z = sigmoid.(x .- log.(offset)) 
  idx = [d < dims ? (1:size_x[d]) : 1:size_x[d] - 1 for d in 1:dims]
  detJ = sum(log.(1 .- z) .+ log.(getindex(p, idx...)), dims=dims)

  return detJ
end

#= TEST
x = randn(3,5,2)
p = sbt(x)
sbt_logabsdetJ(x, p)
=#

