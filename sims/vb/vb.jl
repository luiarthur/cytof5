using BSON
using Cytof5.VB

# For BSON
using Flux, Distributions, RCall

# Load sim data
SIMDAT_PATH = "../sim_study/simdata/kills-flowsom/N500/K5/90/simdat.bson"
simdat = BSON.load(SIMDAT_PATH)[:simdat]

VB.fit
