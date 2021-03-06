println("Loading Libraries onto main node ...")
include("post_process_imports.jl")
println("Finished loading Libraries onto main node ...")

using Distributed
addprocs(20)

@everywhere include("post_process_defs.jl")
include("../Util/Util.jl")

if length(ARGS) > 0
  RESULTS_DIR = ARGS[1]
  AWS_SIM_BUCKET = ARGS[2]
else
  RESULTS_DIR = "results/test-sims"
  AWS_SIM_BUCKET = "s3://cytof-repfam/test-sims"
end

# Collecting paths to output
paths_to_output = ["$(root)/output.bson"
                   for (root, dirs, files) in walkdir(RESULTS_DIR)
                   if "output.bson" in files]

# Post processing
println("Doing post processing...")
@everywhere function makeplots(path_to_output)
  println("Processing: $(path_to_output)")
  pathdir = getpath(path_to_output)
  path_to_simdat = "$(pathdir)/simdat.bson"
  post_process(path_to_output, path_to_simdat=path_to_simdat)
end

println("Make graphs in parallel ...")
status = pmap(makeplots, paths_to_output, on_error=identity)

println("Send results to S3 ...")
Util.s3sync(RESULTS_DIR, AWS_SIM_BUCKET)

println("Success / Failure status: ")
simsucceeded(x) = (x == nothing)
simfailed(x) = !simsucceeded(x)
numsuccess = count(simsucceeded, status)
println("Number of simulations successfully processed: $(numsuccess)")
failures_indices = findall(simfailed, status)
println("Simulations unsuccessfully processed:")
foreach(i -> println(paths_to_output[i]), failures_indices)


println("DONE!")

#= TEST
path_to_output = "results/test-sims/KMCMC2/z1/scale0/output.bson"
path_to_simdat = "results/test-sims/KMCMC2/z1/scale0/simdat.bson"
include("post_process_defs.jl")
post_process(path_to_output, path_to_simdat=path_to_simdat)
=#
