println("pid: $(getpid())"); flush(stdout)

using BSON
using Cytof5

if length(ARGS) == 0
  SEED = 0
  RESULTS_DIR = "results/vb-sim-paper/test/$(SEED)/"
  SIMDAT_PATH = "../sim_study/simdata/kills-flowsom/N5000/K10/1/simdat.bson"
  K_VB = 30
  BATCHSIZE = 2000
else
  SEED = parse(Int, ARGS[1])
  RESULTS_DIR = ARGS[2]
  SIMDAT_PATH = ARGS[3]
  K_VB = parse(Int, ARGS[4])
  BATCHSIZE = parse(Int, ARGS[5])
end
mkpath(RESULTS_DIR)

# For BSON
using Flux, Distributions

# Load sim data
# SIMDAT_PATH = "../sim_study/simdata/kills-flowsom/N500/K5/90/simdat.bson"
# SIMDAT_PATH = "../sim_study/simdata/kills-flowsom/N5000/K10/1/simdat.bson"
simdat = BSON.load(SIMDAT_PATH)[:simdat]
simdat[:y] = Vector{Matrix{Float64}}(simdat[:y])

# Generate model constnats
c = Cytof5.VB.Constants(y=simdat[:y], K=K_VB, L=Dict(false=>5, true=>3),
                        # yQuantiles=[.0, .25, .5], pBounds=[.05, .8, .05],
                        yBounds=[-6., -4., -2.], pBounds=[.05, .8, .05],
                        use_stickbreak=false, tau=.005)

println("seed: $SEED")
# Fit model
out = Cytof5.VB.fit(y=simdat[:y], niters=20000, batchsize=BATCHSIZE, c=c,
                    nsave=30, seed=SEED, flushOutput=true)

# Save results
out[:simdat] = simdat
BSON.bson("$(RESULTS_DIR)/output.bson", out)

# Post process
include("post_process_defs.jl")
out = nothing
post_process("$(RESULTS_DIR)/output.bson")
