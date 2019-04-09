using Cytof5
using RCall, Distributions, Random
using BSON

@rimport xtable
@rimport FlowSOM
@rimport flowCore
@rimport rcommon
@rimport cytof3

SIMDAT_PATH = ARGS[1]
RESULTS_DIR = ARGS[2]
if length(ARGS) == 0
  # Where to get data
  SIMDAT_PATH = ""
  # Where to put results from FlowSOM analysis
  RESULTS_DIR = "results/flowsom/"
else
  SIMDIR = ARGS[1]
  RESULTS_DIR = ARGS[2]
end

# mkpath(OUTPUT_DIR)
replaceMissing(yi, x) = (yi[isnan.(out)] .= x)
