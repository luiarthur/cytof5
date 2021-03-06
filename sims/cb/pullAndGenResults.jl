include("post_process_defs.jl")

### Parse Args ###

# Get
if length(ARGS) >= 1
  AWS_BUCKET = ARGS[1]
else
  println("usage: julia pullAndGenResults.jl <AWS_BUCKET>")
  exit(1)
end

# Create dir to do post processing
TMP_DIR = "TMP_DIR/"
mkpath(TMP_DIR)

RESULTS_DIR = split(read(`aws s3 ls $(AWS_BUCKET)/`, String))
RESULTS_DIR = filter(d -> d != "PRE", RESULTS_DIR)
RESULTS_DIR = filter(d -> !occursin("EXCLUDE_ME", d), RESULTS_DIR)
# RESULTS_DIR = filter(d -> occursin("K21", d) || occursin("missmech", d), RESULTS_DIR)

successes = []
failures = []

@time for dir in sort(RESULTS_DIR, rev=true)
  println("Processing: $dir")
  aws_dir = "$(AWS_BUCKET)/$(dir)"
  tmp_dir = "$(TMP_DIR)/$(dir)"

  try 
    # Pull the dataset to TMP_DIR
    println("Pull data set from AWS to $(tmp_dir)...")
    cmd = `aws s3 sync $aws_dir $tmp_dir`
    println(cmd)
    run(cmd)

    # Post process
    println("Post processing...")
    path_to_mcmc = "$(tmp_dir)/output.jld2"
    @time post_process(path_to_mcmc)

    # Send back to AWS
    println("Sending results to S3...")
    run(`aws s3 sync $tmp_dir $aws_dir`)

    # Remove all files
    println("Remove results locally...")
    run(`rm -rf $(tmp_dir)`)

    append!(successes, [tmp_dir])
  catch
    append!(failures, [tmp_dir])
  end
end


println("Successes")
[println(s) for s in successes];
println()

println("Failures")
[println(f) for f in failures];
println()

