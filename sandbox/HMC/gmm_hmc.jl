include("HMC.jl")
using Distributions, Random
using Flux, Flux.Tracker
using Flux.Tracker: @grad
include("../../src/VB/ADVI/custom_grads.jl")
include("../../src/VB/ADVI/custom_functions.jl")
include("../../src/VB/ADVI/StickBreak.jl")

using RCall
@rimport graphics as rgraphics
@rimport rcommon

struct State
  mu
  log_sig2
  stickbreak_w
end

pretty(s::State) = "State(μ => $(s.mu); σ² => $(s.log_sig2[1]))))"

State(K::Int) = State(param(zeros(K)), param(zeros(K)), param(zeros(K - 1)))

# TODO
