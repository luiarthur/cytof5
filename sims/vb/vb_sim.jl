println("pid: $(getpid())"); flush(stdout)
include("sample/util.jl") # compress_simdat!

using BSON
using Cytof5

if length(ARGS) == 0
  SEED = 0
  RESULTS_DIR = "results/test/vb-sim-paper/$(SEED)/"
  SIMDAT_PATH = "../sim_study/simdata/kills-flowsom/N5000/K10/1/simdat.bson"
  K_VB = 30
  BATCHSIZE = 100
  NITERS = 5000
else
  SEED = parse(Int, ARGS[1])
  RESULTS_DIR = ARGS[2]
  SIMDAT_PATH = ARGS[3]
  K_VB = parse(Int, ARGS[4])
  BATCHSIZE = parse(Int, ARGS[5])
  NITERS = 20000
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
c = Cytof5.VB.Constants(y=simdat[:y], K=K_VB, L=Dict(false=>5, true=>5),
                        yQuantiles=[.0, .25, .5], pBounds=[.05, .8, .05],
                        use_stickbreak=false, tau=.005)
c.priors.eps = Beta(1, 99)
c.priors.sig2 = LogNormal(log(.3), .2)

println("seed: $SEED")
# Fit model
out = Cytof5.VB.fit(y=simdat[:y], niters=NITERS, batchsize=BATCHSIZE, c=c,
                    nsave=30, seed=SEED, flushOutput=true)

# Save results
out[:simdat] = compress_simdat!(simdat)
BSON.bson("$(RESULTS_DIR)/output.bson", out)

# Post process
include("post_process_defs.jl")
out = nothing
post_process("$(RESULTS_DIR)/output.bson")
