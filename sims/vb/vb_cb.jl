println("pid: $(getpid())"); flush(stdout)

using BSON
using Cytof5
include("../cb/PreProcess.jl")

if length(ARGS) == 0
  SEED = 1
  RESULTS_DIR = "results/test/vb-cb-paper/"
  DATA_PATH="../cb/data/cytof_cb_float32.bson"
  K_VB = 30
  BATCHSIZE = 500
  NITERS = 50000
else
  SEED = parse(Int, ARGS[1])
  RESULTS_DIR = ARGS[2]
  DATA_PATH = ARGS[3]
  K_VB = parse(Int, ARGS[4])
  BATCHSIZE = parse(Int, ARGS[5])
  NITERS = 20000
end
mkpath(RESULTS_DIR)

# For BSON
using Flux, Distributions

# Load cb data
y = Matrix{Float64}.(BSON.load(DATA_PATH)[:y])
goodColumns, J = PreProcess.preprocess!(y, maxNanOrNegProp=.9, maxPosProp=.9,
                                        subsample=0.0, rowThresh=-6.0)
Cytof5.Model.logger("good columns: $goodColumns")


# Generate model constnats
c = Cytof5.VB.Constants(y=y, K=K_VB, L=Dict(false=>5, true=>3),
                        yQuantiles=[.0, .25, .5], pBounds=[.05, .8, .05],
                        use_stickbreak=false, tau=.001)
c.priors.eps = Beta(1, 99)
# c.priors.sig2 = LogNormal(log(.3), .2)
# c.priors.sig2 = Gamma(2, 1/20)
c.priors.sig2 = Gamma(.1, 1)

println("seed: $SEED")
# Fit model
out = Cytof5.VB.fit(y=y, niters=NITERS, batchsize=BATCHSIZE, c=c,
                    nsave=30, seed=SEED, flushOutput=true)

# Save results
out[:y] = Matrix{Float32}.(y)
BSON.bson("$(RESULTS_DIR)/output.bson", out)

# Post process
include("post_process_defs.jl")
out = nothing
post_process("$(RESULTS_DIR)/output.bson")
