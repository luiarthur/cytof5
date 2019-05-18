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

# implement for normal, gamma, invgamma, dirichlet, beta

import SpecialFunctions: digamma

"""
gradient of log of normal density with respect to μ and log(σ)
"""
function qgrad_normal(x, m, log_s)
  s = exp(log_s)
  z = (x - m) / s
  # Gradient wrt μ
  grad_mu = z / s
  # Gradient wrt log(σ)
  grad_log_s = -1 + z ^ 2

  return grad_mu, grad_log_s
end

"""
gradient of log of gamma density with respect to log(shape) and log(scale)
"""
function qgrad_gamma(x, log_shape, log_scale)
  a = exp(log_shape)
  b = exp(log_scale)
  grad_log_shape = -digamma(a) * a - a * log_scale + a * log(x)
  grad_log_scale = -a + (x / b)

  return grad_log_shape, grad_log_scale
end


