include("post_process_defs.jl")

function lsRec(path)
  return split(read(`find $path`, String))
end

function dirContains(word, dir)::Bool
  allFiles = lsRec(dir)
  for f in allFiles
    if occursin(word, f)
      return true
    end
  end
  return false
end

### MAIN ###
RESULTS_DIR = length(ARGS) >= 1 ? ARGS[1] : "results/cb_preprocessed/"
MCMC_OUTPUT = filter(d -> occursin(".jld2", d), lsRec(RESULTS_DIR))
MCMC_OUTPUT = filter(d -> !occursin("reduced_cb.jld2", d), MCMC_OUTPUT)
println(MCMC_OUTPUT)
successes = []
failures = []

@time for mcmc in MCMC_OUTPUT
  println("Processing: $mcmc")
  try 
    datapath = 
    @time post_process(mcmc)
    append!(successes, [mcmc])
  catch
    append!(failures, [mcmc])
  end
end

println("Successes")
[println(s) for s in successes];
println()

println("Failures")
[println(f) for f in failures];
println()

