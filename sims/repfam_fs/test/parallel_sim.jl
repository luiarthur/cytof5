# TODO: Make a parallel simulation file.
# - Call with 
# julia  parallel_runner.jl  config.jl  aws_sim_bucket  sim_results_dir
# - Put all config.jl in sim-config/

using Distributed

# NOTE: Change these
SIMDIR = "configs/test-sim"

@everywhere include("../Util/Util.jl")

function setnumcores(n::Int)
  children_procs = filter(w -> w > 1, workers())
  rmprocs(children_procs)
  addprocs(n)
end

# NOTE: read this from configs/test-sim-x-y.jl
include("$(SIMDIR)/settings.jl")
settings = Settings.settings
ncores = min(20, length(settings))  # use at most 20 cores on server
setnumcores(ncores)

# NOTE: change this
path_to_simfn = "$(SIMDIR)/simfn.jl"
@everywhere include(path_to_simfn)

@everywhere function simfn(setting)
  results_dir = setting[:results_dir]
  mkpath(results_dir)
  println("Running $(results_dir)")
  flush(stdout)
  Util.redirect_all("$(results_dir)/log") do
    Sim.simfn(setting)
  end
end

# NOTE:
# - f::Function: A function which takes one argument of type Dict{Any}
# - settings::Vector{Dict}): A vector of settings
result = pmap(simfn, settings, on_error=identity);
println(result)
