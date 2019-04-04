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
z = sum(log.(p))
back!(z)
@assert all(abs.(y.tracker.grad - [-0.9015, -0.7284, -0.4621]) .< 1e-4)
# 3-element Array{Float64,1}:
# -0.9014675456746869
# -0.7283506542974874
# -0.4621171572600099


#= Test
# TODO: minus example with prints of Δ
using Flux.Tracker: TrackedArray, track, @grad
mult(a, b) = a .* b
mult(a::TrackedArray, b::TrackedArray) = track(mult, a, b)
@grad function mult(a, b)
  return mult(a.data, b.data), function(Δ)
    println(Δ)
    x = (Δ .* b.data, a ./ Δ)
    println(x)
    return x
  end
end
a = param([1, 2, 3])
b = param([3, 2, 1])
back!(sum(mult(a, b) .^ 2))
a.grad
b.grad
=#
