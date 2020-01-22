# TODO: Make a parallel simulation file.
# - Call with 
# julia  parallel_runner.jl  config.jl  aws_sim_bucket  sim_results_dir
# - Put all config.jl in sim-config/

using Distributed

# NOTE: Change this.
if length(ARGS) == 0
  SIMDIR = "configs/test-sim"
else
  SIMDIR = ARGS[1]
end

function setnumcores(n::Int)
  children_procs = filter(w -> w > 1, workers())
  rmprocs(children_procs)
  addprocs(n)
end

# NOTE: read configs
include("$(SIMDIR)/settings.jl")
settings = Settings.settings
ncores = min(20, length(settings))  # use at most 20 cores on server
setnumcores(ncores)

@everywhere SIMDIR = $SIMDIR
@everywhere include("../Util/Util.jl")
@everywhere include("$(SIMDIR)/simfn.jl")

@everywhere function simfn(setting)
  results_dir = setting[:results_dir]
  aws_bucket = setting[:aws_bucket]
  mkpath(results_dir)

  println("Running $(results_dir)")
  flush(stdout)

  Util.redirect_all("$(results_dir)/log") do
    Sim.simfn(setting)
  end

  # Send to S3.
  Util.s3sync(results_dir, aws_bucket, `--exclude '*.nfs'`)

  # Remove results to save space.
  rm(results_dir, recursive=true)

  return
end

# NOTE:
# - f::Function: A function which takes one argument of type Dict{Any}
# - settings::Vector{Dict}): A vector of settings
result = pmap(simfn, settings, on_error=identity);
println(result)
