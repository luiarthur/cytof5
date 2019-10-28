println("Loading Libraries...")
include("post_process_defs.jl")
include("../Util/Util.jl")

if length(ARGS) > 0
  RESULTS_DIR = ARGS[1]
  AWS_SIM_BUCKET = ARGS[2]
else
  RESULTS_DIR = "results/test-sims"
  AWS_SIM_BUCKET = "s3://cytof-repfam/test-sims"
end

# Collecting paths to output
paths_to_output = String[]
for (root, dirs, files) in walkdir(RESULTS_DIR)
  if "output.bson" in files
    append!(paths_to_output, ["$(root)/output.bson"])
  end
end

# Post processing
println("Doing post processing...")
for path_to_output in paths_to_output
  println("Processing: $(path_to_output)")
  pathdir = getpath(path_to_output)
  path_to_simdat = "$(pathdir)/simdat.bson"
  post_process(path_to_output, path_to_simdat)

end

Util.s3sync(RESULTS_DIR, AWS_SIM_BUCKET)
println("DONE!")

#= TEST
path_to_output = "results/test-sims/KMCMC2/z1/scale0/output.bson"
path_to_simdat = "results/test-sims/KMCMC2/z1/scale0/simdat.bson"
include("post_process_defs.jl")
post_process(path_to_output, path_to_simdat=path_to_simdat)
=#
