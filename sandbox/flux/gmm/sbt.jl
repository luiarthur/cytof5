function sbt(x)
  K = length(x) 
  offset = (K + 1) .- cumsum(ones(K), dims=1)
  z = sigmoid.(x .- log.(offset)) 
  z_cumprod = cumprod(1 .- z, dims=1)
  # y = cat(z, pad, dims=dims) .* cat(pad, z_cumprod, dims=1)
  y = vcat(z, 1) .* vcat(1, z_cumprod)

  return y
end

function sbt_inv(p)
  # see: https://pytorch.org/docs/stable/_modules/torch/distributions/transforms.html
  error("unimplemented")
end

function sbt_logabsdetJ(x, p)
  K = length(x)

  offset = (K + 1) .- cumsum(ones(K), dims=1)
  z = sigmoid.(x .- log.(offset)) 
  detJ = sum(log.(1 .- z) .+ log.(p[1:end-1]))

  return detJ
end

#= TEST
x = param(randn(3))
p = sbt(x)
sbt_logabsdetJ(x, p)
logpdf(Dirichlet(ones(4)), p) + sbt_logabsdetJ(x, p)
=#

