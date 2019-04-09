using Cytof5
using RCall, Distributions, Random
using BSON

@rimport xtable
@rimport FlowSOM
@rimport flowCore
@rimport rcommon
@rimport cytof3

if length(ARGS) == 0
  # Where to get data
  SIMDAT_PATH = "results/sim-paper/sim_Nfac500_K5_KMCMC02/output.bson"
  # Where to put results from FlowSOM analysis
  RESULTS_DIR = "results/flowsom/"
else
  SIMDAT_PATH = ARGS[1]
  RESULTS_DIR = ARGS[2]
end

# TODO:
# mkpath(OUTPUT_DIR)

# Load simulated data
@time y = BSON.load(SIMDAT_PATH)[:simdat][:y]

# replace missing values with -20
replaceMissing(yi, x) = (yi[isnan.(yi)] .= x)
for yi in y; replaceMissing(yi, -20); end

# Combine into one matrix Y
Y = vcat(y...)
J = size(Y, 2)
@rput Y

# Create flowframe
ff_Y = R"colnames(Y) <- 1:NCOL(Y); flowCore::flowFrame(Y)"
@time fSOM = FlowSOM.FlowSOM(ff_Y,
                             # Input options:
                             colsToUse = 1:J,
                             # Metaclustering options:
                             #nClus = 20,
                             maxMeta=20,
                             # Seed for reproducible results:
                             seed = 42)



