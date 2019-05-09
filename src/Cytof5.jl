module Cytof5
#= Note:
Julia uses Gamma(shape, scale) and InverseGamma(shape, scale).
Note that InverseGamma(shape, scale) IS my preferred parameterization.
It has a mean of scale / (shape - 1) for shape > 1.
=#

export MCMC

include("Util/Util.jl")
include("MCMC/MCMC.jl")
include("Model/Model.jl")
include("VB/VB.jl")

# TODO: get rid of RCall, use Plots with GR backend.
# include("Plot/Plot.jl")

# For R plots
include("R/Plots.jl")

end # module
