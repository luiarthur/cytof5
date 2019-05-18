include("grad.jl")

using Flux, Flux.Tracker, Distributions
import Random

Random.seed!(0)


# Normal
m = param(1)
log_s = param(log(.5))
# x = 2.0
z = rand()
x = Tracker.data(m + z * exp(log_s))

@time back!(logpdf(Normal(m, exp(log_s)), x))
@time res = qgrad(x, m.data, log_s.data)
@assert (m.tracker.grad, log_s.tracker.grad) == res

B = Int(1e5)
@time for i in 1:B
  back!(logpdf(Normal(m, exp(log_s)), x))
end
@time for i in 1:B
  res = qgrad_normal(x, m.data, log_s.data)
end
# @assert (m.tracker.grad, log_s.tracker.grad) == res


# Gamma
log_a = param(log(1))
log_b = param(log(2))
a = exp(log_a)
b = exp(log_b)
x = rand(Gamma(a, b)).data

back!(logpdf(Gamma(a, b), x))
res_ad = (log_a.tracker.grad, log_b.tracker.grad)
res = qgrad_gamma(x, log_a.data, log_b.data)
@assert res_ad == res
