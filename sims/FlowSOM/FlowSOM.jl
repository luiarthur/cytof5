import Pkg
Pkg.activate("../../")

using Cytof5
using RCall, Distributions, Random
using JLD2, FileIO

R"""
library(xtable)
library(FlowSOM)
library(flowCore)
library(rcommon)
library(cytof3)
source("est_Z_from_clusters.R")
set.seed(3)
"""

include("../sim_study/util.jl")

# Where to put results from FlowSOM analysis
OUTDIR = "results/"

# Where to get data
if length(ARGS) == 0
  SIMDIR = "../sim_study/results/I3_J32_N_factor100_K8_L4_K_MCMC10_L_MCMC5_b0PriorSd0.1_b1PriorScale0.1_SEED0/"
else
  SIMDIR = ARGS[1]
end

# READ DATA
println("Loading Data ...")
@load "$(SIMDIR)/output.jld2" out dat ll lastState c y_dat

# MAIN
function replaceMissing(yi, x)
  out = deepcopy(yi)
  out[ismissing.(out)] .= x
  return out
end

# Preimpute missing values
y_tilde = [ replaceMissing(yi, -20) .+ 0 for yi in y_dat.y ]

# Convert data to flowsom formats
Y_tilde = R"Reduce(rbind, $y_tilde)"
@rput Y_tilde
ff_Y = R"colnames(Y_tilde) <- 1:NCOL(Y_tilde); flowFrame(Y_tilde)"

# FlowSOM
println("Running FlowSOM...")
J = y_dat.J
I = y_dat.I
FlowSOM = R"FlowSOM::FlowSOM"
@time fSOM = FlowSOM(ff_Y,
                     # Input options:
                     colsToUse = 1:J,
                     # Metaclustering options:
                     #nClus = 20,
                     maxMeta=20,
                     # Seed for reproducible results:
                     seed = 42)

@rput fSOM
R"""
fSOMClus = fSOM$meta[fSOM$FlowSOM$map$mapping[,1]]
fsEst = as.list(1:$I)
fsClus = as.numeric(fSOMClus)

mult=1
for (i in 1:I) {
  clus = fsClus[idx[i,1]:idx[i,2]]
  print(length(unique(clus))) # Number of clusters learned
  #est = est_ZW_from_clusters(y_tilde[[i]], clus, f=median)
  #yZ(yi=y[[i]], Zi=est$Z*1, Wi=est$W, cell_types_i=est$clus-1,
  #   zlim=c(-3,3), na.color='black', thresh=.9, col=blueToRed(7),
  #   cex.z.b=1.5, cex.z.lab=1.5, cex.z.l=1.5, cex.z.r=1.5,
  #   cex.y.ylab=1.5, cex.y.xaxs=1.4, cex.y.yaxs=1.4, cex.y.leg=1.5, 
  #   prop_lower_panel=0)
  clus = relabel_clusters(clus)
  my.image(y[[i]][order(clus),], col=blueToRed(11), zlim=c(-5,5), addL=TRUE,
           na.color='black', cex.y.leg=2, xlab='cell types',  ylab='cells',
           cex.lab=1.5, cex.axis=1.5, xaxt='n',
           f=function(z) {
             add.cut(clus) 
             axis(1, at=1:J, fg='grey', las=2, cex.axis=1.5)
           })
}
"""

