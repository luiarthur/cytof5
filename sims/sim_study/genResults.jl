#!/usr/bin/env julia
include("post_process_defs.jl")

function isDir(path)
  ftype = strip(split(read(`file $path`, String), ":")[2])
  return ftype == "directory"
end

function lsRec(path)
  return split(read(`find $path`, String))
end

#= Test
lsRec("results/sim3")
=#

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
RESULTS_DIR = length(ARGS) >= 1 ? ARGS[1] : "results/sim3/"
MCMC_OUTPUT = filter(d -> occursin(".jld2", d), lsRec(RESULTS_DIR))
# TODO: make retrieve4
# Then make postprocess4 with this extra line
#MCMC_OUTPUT = filter(d -> occursin("N_factor10000", d), MCMC_OUTPUT) # Run this additional filter
successes = []
failures = []

@time for mcmc in MCMC_OUTPUT
  println("Processing: $mcmc")
  try 
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

