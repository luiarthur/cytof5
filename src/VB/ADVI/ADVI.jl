module ADVI

using Distributions
using Flux
using Flux.Tracker
using Flux.Tracker: TrackedArray, @grad, track

export MPA, MPR, rsample, vp, ModelParam, logabsdetJ, transform  # ModelParam.jl


# Custom gradient definitions
include("custom_grads.jl")

# Stick breaking transform of simplexes
include("StickBreak.jl")

# Model Parameters
include("ModelParam.jl")
MPA{F, N} = ModelParam{TrackedArray{F, N, Array{F, N}}, F, NTuple{N, Int}}
MPR{F} = ModelParam{Tracker.TrackedReal{F}, F, NTuple{0, Int}}

end
