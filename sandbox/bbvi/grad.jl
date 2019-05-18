#= Paper
BBVI:
http://proceedings.mlr.press/v33/ranganath14.pdf

Supplementary:
http://proceedings.mlr.press/v33/ranganath14-supp.pdf
=#

"""
gradient of log of normal density with respect to μ and log(σ)
"""
function qgrad(x, m, log_s)
  s = exp(log_s)
  # Gradient wrt μ
  grad_mu = (x - m) / s^2
  # Gradient wrt log(σ)
  grad_log_s = -1 + ((x - m) / s)^2

  return grad_mu, grad_log_s
end


#= Test
using Flux, Flux.Tracker, Distributions
m = param(1)
log_s = param(log(.5))
x = 2

@time back!(logpdf(Normal(m, exp(log_s)), x))
@time res = qgrad(x, m.data, log_s.data)
(m.tracker.grad, log_s.tracker.grad) == res
=#
