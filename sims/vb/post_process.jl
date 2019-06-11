println("Loading Libraries...")
include("post_process_defs_pyplot.jl")

# RESULTS_DIR = "results/vb-sim-paper/"
RESULTS_DIR = ARGS[1]

# Collecting paths to output
paths_to_output = String[]
for (root, dirs, files) in walkdir(RESULTS_DIR)
  if "output.bson" in files
    append!(paths_to_output, ["$root/output.bson"])
  end
end

# Post processing
println("Doing post processing...")
for path_to_output in paths_to_output
  println("Processing: $(path_to_output)")
  post_process(path_to_output)
end
