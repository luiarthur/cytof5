module Model

using Distributions

# TODO: Remove this dependency
using RCall # Mclust

import LinearAlgebra # Identity matrix
import Random # shuffle
import StatsBase # wsample, counts

include("../MCMC/MCMC.jl")
import .MCMC.Util.@namedargs

include("util.jl")
include("State.jl")
include("Data.jl")
include("Constants.jl")
include("Tuners.jl")
include("update.jl")
include("repFAM/repFAM.jl")
include("FeatureSelect/FeatureSelect.jl")
include("genInitialState.jl")
include("fit.jl")

#=
precompile(cytof5_fit, (State, Constants, Data, Int, Int,
                        Vector{Vector{Symbol}}, Vector{Int}, Bool, Int, Bool,
                        Bool, Bool))
=#

end # Model
