# TODO: Make a parallel simulation file.
# - Call with 
# julia  parallel_runner.jl  config.jl  aws_sim_bucket  sim_results_dir
# - Put all config.jl in sim-config/

using Distributed

function setnumcores(n::Int)
  children_procs = filter(w -> w > 1, workers())
  rmprocs(children_procs)
  addprocs(n)
end

setnumcores(3)  # TODO: change this to 20 on servers

# TODO: change this
@everywhere include("configs/test_thing.jl")

# TODO: read this from configs/test-sim-x-y.jl
settings = 1:8

# TODO: change the arguments to 
# - f::Function: A function which takes one argument of type Dict{Any}
# - settings::Vector{Dict}): A vector of settings

result = pmap(Simulation.f, settings, on_error=identity);
