include("model.jl")
using Flux, Flux.Tracker

# Generate Data
N = 1000
x = randn(N)
b0 = 2.0
b1 = -3.0
sig = 0.5
y = b0 .+ b1 .* x .+ sig

vps = Dict(:b0 => VarParam(), :b1 => VarParam(), :sig => VarParam())
loss(y, x, vps) = -elbo(y, x, vps)

grads = Tracker.gradient(vps -> loss(x, y, vps), param(vps))
opt = ADAM(.1)

Tracker.update!(opt, grads)

for p in params
  Tracker.update!(opt, p, grads[p])
end
