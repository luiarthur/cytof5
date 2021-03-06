module ADVI

using Distributions
using Flux
using Flux.Tracker
using Flux.Tracker: TrackedArray, @grad, track, lgamma

export MPA, MPR, rsample, vp, ModelParam, logabsdetJ, transform  # ModelParam.jl


# Custom gradient definitions
include("custom_grads.jl")

# Custom functions
include("custom_functions.jl")

# Stick breaking transform of simplexes
include("StickBreak.jl")

# Model Parameters
include("ModelParam.jl")
MPA{F, N} = ModelParam{TrackedArray{F, N, Array{F, N}}, NTuple{N, Int}}
MPR{F} = ModelParam{Tracker.TrackedReal{F}, NTuple{0, Int}}

end
