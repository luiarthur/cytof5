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
RESULTS_DIR = "results/"

# Where to get data
if length(ARGS) == 0
  SIMDIR = "../sim_study/results/I3_J32_N_factor100_K8_L4_K_MCMC10_L_MCMC5_b0PriorSd0.1_b1PriorScale0.1_SEED0/"
else
  SIMDIR = ARGS[1]
end

# Where in results dir to put output
expName(dir) = filter(x -> length(x) > 0, split(dir, "/"))[end]
OUTPUT_DIR = "$(RESULTS_DIR)/$(expName(SIMDIR))/"
mkpath(OUTPUT_DIR)

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
N = y_dat.N
FlowSOM = R"FlowSOM::FlowSOM"
@time fSOM = FlowSOM(ff_Y,
                     # Input options:
                     colsToUse = 1:J,
                     # Metaclustering options:
                     #nClus = 20,
                     maxMeta=20,
                     # Seed for reproducible results:
                     seed = 42)

@rput fSOM I J N
R"""
idx_upper = cumsum(N)
idx_lower = c(1,idx_upper[-I]+1)
idx = cbind(idx_lower, idx_upper)

fSOMClus = fSOM$meta[fSOM$FlowSOM$map$mapping[,1]]
fsEst = as.list(1:$I)
fsClus = as.numeric(fSOMClus)

mult=1
png($OUTPUT_DIR %+% 'YZ%03d_FlowSOM.png', height=500*mult, width=500*mult)
for (i in 1:$I) {
  clus = fsClus[idx[i,1]:idx[i,2]]
  print(length(unique(clus))) # Number of clusters learned
  clus = relabel_clusters(clus)
  my.image($(y_dat.y)[[i]][order(clus),], col=blueToRed(11), zlim=c(-5,5), addL=TRUE,
           na.color='black', cex.y.leg=2, xlab='cell types',  ylab='cells',
           cex.lab=1.5, cex.axis=1.5, xaxt='n',
           f=function(z) {
             add.cut(clus) 
             axis(1, at=1:$(J), fg='grey', las=2, cex.axis=1.5)
           })
}
dev.off()

println("Computing ARI ...")
true.clus.ls <- $(dat[:lam])
fs.clus.ls = lapply(1:I, function(i) fsClus[idx_lower[i]:idx_upper[i]])

ARI = sapply(1:I, function(i) {
  ari(true.clus.ls[[i]], fs.clus.ls[[i]])
})

ARI_all = 'FlowSOM'=ari(unlist(true.clus.ls), fsClus)

sink($OUTPUT_DIR %+% "ari.txt")
  println("ARI:")
  print(ARI)
  println("ARI all:")
  print(ARI_all)
sink()
"""

