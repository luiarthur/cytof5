include("grad.jl")

using Flux, Flux.Tracker, Distributions
import Flux.Tracker: lgamma
import Random

Random.seed!(0)


# Normal
println("Test Normal")
m = param(1)
log_s = param(log(.5))
# x = 2.0
z = rand()
x = Tracker.data(m + z * exp(log_s))

@time back!(logpdf(Normal(m, exp(log_s)), x))
@time res = grad_logq_normal(x, m.data, log_s.data)
@assert (m.tracker.grad, log_s.tracker.grad) == res

B = Int(1e5)
@time for i in 1:B
  back!(logpdf(Normal(m, exp(log_s)), x))
end
@time for i in 1:B
  res = grad_logq_normal(x, m.data, log_s.data)
end
# @assert (m.tracker.grad, log_s.tracker.grad) == res


# Gamma
println("Test Gamma")
log_a = param(log(1))
log_b = param(log(2))
a = exp(log_a)
b = exp(log_b)
x = rand(Gamma(a.data, b.data))

back!(logpdf(Gamma(a, b), x))
res_ad = (log_a.tracker.grad, log_b.tracker.grad)
res = grad_logq_gamma(x, log_a.data, log_b.data)
@assert res_ad == res


# Beta
println("Test Beta")
log_a = param(log(1))
log_b = param(log(2))
a = exp(log_a)
b = exp(log_b)
x = rand(Beta(a.data, b.data))

back!(logpdf(Beta(a, b), x))
res_ad = (log_a.tracker.grad, log_b.tracker.grad)
res = grad_logq_beta(x, log_a.data, log_b.data)
@assert res_ad == res

# Dirichlet
println("Test Dirichlet")
lpdf_dirichlet(p, a) = lgamma(sum(a)) - sum(lgamma.(a)) + sum((a .- 1) .* log.(p))
log_alpha = param(rand(10))
alpha = exp.(log_alpha)
x = rand(Dirichlet(Tracker.data(alpha)))

back!(lpdf_dirichlet(x, alpha))
res_ad = log_alpha.grad
res = grad_logq_dirichlet(x, log_alpha.data)
@assert all(abs.(res_ad - res) .< 1e-6)
