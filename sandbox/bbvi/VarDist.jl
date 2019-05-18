module Dev

using Distributions
include("grad.jl")

mutable struct VDNormal{F <: AbstractFloat}
  m::F
  s::F
  log_s::F
  VDNormal(m::F, s::F) where F = new{F}(m, s, log(s))
end

rand(d::VDNormal) = randn() * d.s + d.m
grad_logq(d::VDNormal, x) = grad_logq_normal(d.m, d.log_s, x)

function update!(d::VDNormal, grads, lr)
  g_m, g_log_s = grads
  d.m += lr * g_m
  d.log_s += lr * g_log_s
  d.s = exp(d.log_s)
  return d
end

end # Dev

#= Test
vd = Dev.VDNormal(1., .5)
x = Dev.rand(vd)
grads = Dev.grad_logq(vd, x)
Dev.update!(vd, grads, .01)
=#
