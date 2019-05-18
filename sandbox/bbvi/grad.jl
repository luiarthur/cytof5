#= Paper
BBVI:
http://proceedings.mlr.press/v33/ranganath14.pdf

Supplementary:
http://proceedings.mlr.press/v33/ranganath14-supp.pdf

# Another useful resource:
http://www.it.uu.se/research/systems_and_control/education/2018/pml/lectures/VILectuteNotesPart3.pdf

This gradient is for the score function estimator. I don't
understand why the gradioent of the sampled qauntity (x)
is not needed. But that's how it is.
=#

"""
gradient of log of normal density with respect to μ and log(σ)
"""
function qgrad(x, m, log_s)
  s = exp(log_s)
  z = (x - m) / s
  # Gradient wrt μ
  grad_mu = z / s
  # Gradient wrt log(σ)
  grad_log_s = -1 + z ^ 2

  return grad_mu, grad_log_s
end


using Flux, Flux.Tracker, Distributions
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
  res = qgrad(x, m.data, log_s.data)
end
# @assert (m.tracker.grad, log_s.tracker.grad) == res
