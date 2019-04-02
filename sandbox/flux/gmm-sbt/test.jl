# https://github.com/FluxML/Flux.jl/issues/136
include("StickBreak.jl"); SB = StickBreak

using Flux, Flux.Tracker
using Distributions

x = param([.1, .2, .3])
p = SB.transform(x)
@assert abs(SB.logabsdetJ(x, p) - (-5.5850)) < 1e-4
@assert abs(logpdf(Dirichlet(ones(4)), p) + SB.logabsdetJ(x, p) - (-3.7933)) < 1e-4

# TEST
y = param(ones(3))
p = SB.transform(y)
z = sum(p)
back!(z)
y.tracker.grad
