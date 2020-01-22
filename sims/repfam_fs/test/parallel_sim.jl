# TODO: Make a parallel simulation file.
# - Call with 
# julia  parallel_runner.jl  config.jl  aws_sim_bucket  sim_results_dir
# - Put all config.jl in sim-config/

using Distributed

include("../Util/Util.jl")

function setnumcores(n::Int)
  children_procs = filter(w -> w > 1, workers())
  rmprocs(children_procs)
  addprocs(n)
end

# NOTE: read this from configs/test-sim-x-y.jl
include("configs/test-sim/settings.jl")
settings = Settings.settings
ncores = min(20, length(settings))  # use at most 20 cores on server
setnumcores(ncores)

# NOTE: change this
# @everywhere include("configs/test_thing.jl")
@everywhere include("configs/test-sim/simfunc.jl")

function simfunc(setting)
  Util.redirect_all(Simulation.f, "settings[:results]/log")
end

# TODO: change the arguments to 
# - f::Function: A function which takes one argument of type Dict{Any}
# - settings::Vector{Dict}): A vector of settings

result = pmap(, settings, on_error=identity);
println(result)
