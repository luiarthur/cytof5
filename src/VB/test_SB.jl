# https://github.com/FluxML/Flux.jl/issues/136
include("StickBreak.jl"); SB = StickBreak

using Flux, Flux.Tracker
using Distributions

x = param([.1])
p = SB.transform(x)
SB.logabsdetJ(x, p)



x = param([.1, .2, .3])
p = SB.transform(x)
@assert abs(SB.logabsdetJ(x, p)[1] - (-5.5850)) < 1e-4
@assert abs(logpdf(Dirichlet(ones(4)), p) + SB.logabsdetJ(x, p)[1] - (-3.7933)) < 1e-4

# TEST
y = param(ones(Float32, 3))
p = SB.transform(y)
z = sum(log.(p))
back!(z)
@assert all(abs.(y.tracker.grad - [-0.9015, -0.7284, -0.4621]) .< 1e-4)
# 3-element Array{Float64,1}:
# -0.9014675456746869
# -0.7283506542974874
# -0.4621171572600099

y = param(ones(Float32, 3))
p = SB.transform(y)
z = SB.logsumexp(p)[1]
back!(z)
y.tracker.grad


x = cumsum(cumsum(cumsum(ones(2,3,4), dims=1), dims=2), dims=3)
x = param(x / 24)
p = SB.transform(x)
z = sum(log.(p))
back!(z)
x.grad
@assert abs(sum(x.grad) - (-4.8753)) < 1e-4

x = cumsum(cumsum(cumsum(ones(2,3,4), dims=1), dims=2), dims=3)
x = param(x / 24)
p = SB.transform(x)
z = SB.logsumexp(p, dims=3)
back!(sum(z))
x.grad


