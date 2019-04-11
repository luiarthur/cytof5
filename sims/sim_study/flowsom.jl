using Cytof5
using RCall, Distributions, Random
using BSON

@rimport xtable
@rimport FlowSOM
@rimport flowCore
R"source('../FlowSOM/est_Z_from_clusters.R')"

if length(ARGS) == 0
  # Where to get data
  SIMDAT_PATH = "simdata/kills-flowsom/N5000/98/simdat.bson"
  # SIMDAT_PATH = "simdata/kills-flowsom/N500/90/simdat.bson"

  # Where to put results from FlowSOM analysis
  RESULTS_DIR = "results/sim-paper/flowsom/N5000/"
  # RESULTS_DIR = "results/sim-paper/flowsom/N500/"
else
  SIMDAT_PATH = ARGS[1]
  RESULTS_DIR = ARGS[2]
end

# TODO:
mkpath(RESULTS_DIR)

# Load simulated data
@time simdat = BSON.load(SIMDAT_PATH)[:simdat]

# replace missing values with `REPLACEMENT`
REPLACEMENT = -6.0
function replaceMissing(yi, x)
  out = deepcopy(yi)
  out[isnan.(out)] .= x
  return out
end
y = [replaceMissing(yi, REPLACEMENT) for yi in simdat[:y]]

# Combine into one matrix Y
Y = vcat(y...)
J = size(Y, 2)
I = length(y)
N = size.(y, 1)
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
                             seed=42); # high ari for N500

@rput fSOM I J N
R"""
zlim = c(-1, 1) * 4
plotPng = function(fname, s=10, w=480, h=480, ps=12, ...) {
  png(fname, w=w*s, h=h*s, pointsize=ps*s, ...)
}
addCut = function(clus, s=1) abline(h=cumsum(table(clus)) + .5, lwd=3*s, col='yellow')
addGridLines = function(Z, s=1) abline(v=1:NCOL(Z) + .5, h=1:NROW(Z) + .5, col='grey', lwd=1*s)

library(rcommon)
library(cytof3)

idx_upper = cumsum(N)
idx_lower = c(1,idx_upper[-I]+1)
idx = cbind(idx_lower, idx_upper)

fSOMClus = fSOM$meta[fSOM$FlowSOM$map$mapping[,1]]
fsEst = as.list(1:$I)
fsClus = as.numeric(fSOMClus)

mult=1
plotPng($RESULTS_DIR %+% 'YZ%03d_FlowSOM.png', s=10)
for (i in 1:$I) {
  clus = fsClus[idx[i,1]:idx[i,2]]
  print(length(unique(clus))) # Number of clusters learned
  clus = relabel_clusters(clus)
  my.image($(simdat[:y])[[i]][order(clus),], col=blueToRed(9), zlim=zlim, addL=TRUE,
           na.color='black', cex.y.leg=1, xlab='cell types',  ylab='cells',
           cex.lab=1.5, cex.axis=1.5, xaxt='n',
           f=function(z) {
             addCut(clus, s=10)
             axis(1, at=1:$(J), fg='grey', las=2, cex.axis=1.5)
           })
}
dev.off()

println("Computing ARI ...")
true.clus.ls <- $(simdat[:lam])
fs.clus.ls = lapply(1:I, function(i) fsClus[idx_lower[i]:idx_upper[i]])

ARI = sapply(1:I, function(i) {
  ari(true.clus.ls[[i]], fs.clus.ls[[i]])
})

ARI_all = 'FlowSOM'=ari(unlist(true.clus.ls), fsClus)
print(ARI)
print(ARI_all)

sink($RESULTS_DIR %+% "ari.txt")
  println("ARI:")
  print(ARI)
  println("ARI all:")
  print(ARI_all)
sink()
"""


