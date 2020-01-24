# TODO: Make a parallel simulation file.
# - Call with 
# julia  parsim.jl  <path-to-config>
# - Put all config.jl in sim-config/

using Distributed

# NOTE: These libs have to be imported in
#       main process and worker processes.
println("Loading packages. This may take a few minutes ..."); flush(stdout)
using Cytof5  # This is in dev mode
import Pkg; Pkg.activate("../../")  # sims
using Random, Distributions, BSON


# NOTE: Change this.
if length(ARGS) == 3
  SIMDIR = ARGS[1]
  RESULTS_DIR_PREFIX = ARGS[2]
  AWS_BUCKET_PREFIX = ARGS[3]
else
  println("Usage: julia parsim.jl <simdir> <results_dir_prefix> <aws_bucket_prefix>")
end

function setnumcores(n::Int)
  children_procs = filter(w -> w > 1, workers())
  rmprocs(children_procs)
  addprocs(n)
end

# NOTE: read configs
include("$(SIMDIR)/settings.jl")

for setting in settings
  setting[:results_dir] = "$(RESULTS_DIR_PREFIX)/$(setting[:outdir_suffix])"
  setting[:aws_bucket] = "$(AWS_BUCKET_PREFIX)/$(setting[:outdir_suffix])"
end

ncores = min(20, length(settings))  # use at most 20 cores on server
setnumcores(ncores)

println("Sourcing files ..."); flush(stdout)
@everywhere SIMDIR = $SIMDIR
@everywhere include("../Util/Util.jl")
@everywhere include("$(SIMDIR)/simfn.jl")

@everywhere function sim(setting)
  results_dir = setting[:results_dir]
  aws_bucket = setting[:aws_bucket]
  mkpath(results_dir)

  println("Running $(results_dir)")
  flush(stdout)

  Util.redirect_all("$(results_dir)/log") do
    simfn(setting)
  end

  # Send to S3.
  Util.s3sync(results_dir, aws_bucket, `--exclude '*.nfs'`)

  # Remove results to save space.
  # rm(results_dir, recursive=true)
  return
end

# NOTE:
# - f::Function: A function which takes one argument of type Dict{Any}
# - settings::Vector{Dict}): A vector of settings
println("Starting jobs ..."); flush(stdout)
result = pmap(sim, settings, on_error=identity);
println(result)
println("DONE with all runs!")
