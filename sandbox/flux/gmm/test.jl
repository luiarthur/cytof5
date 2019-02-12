include("model.jl")

# Test
K = 10
a = collect(1:K) * 1.0

x = randn(K - 1)
y = param(x)

@time for i in 1:100
  A = lpdf_real_simplex(a, x)
  B = lpdf_real_simplex(a, y)
  @assert abs(A - B) < 1e-6

  C = lpdf_Dirichlet_fullrank(a, softmax_fullrank(x, complete=false))
  D = logpdf(Dirichlet(a), softmax([0.; x]))
  @assert abs(C - D) < 1e-6

  @assert all(abs.(softmax_fullrank(x) .- softmax_safe([0.0; x])) .< 1e-6)

  p = softmax_fullrank(y)
  @assert abs(1 - sum(p)) < 1e-6
end
