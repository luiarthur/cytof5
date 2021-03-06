using Cytof5
using RCall, Distributions, Random
using JLD2, FileIO
include("PreProcess.jl")

function loadSingleObj(objPath)
  data = load(objPath)
  return data[collect(keys(data))[1]]
end

function replaceMissing(yi, x)
  out = deepcopy(yi)
  out[isnan.(out)] .= x
  return out
end


@rimport FlowSOM
@rimport flowCore
@rimport xtable

if length(ARGS) == 0
  DATA_PATH = "data/cytof_cb_with_nan.jld2"
  RESULTS_DIR = "results/cb-paper/flowsom/"
else
  DATA_PATH = ARGS[1]
  RESULTS_DIR = ARGS[2]
end

# TODO:
mkpath(RESULTS_DIR)

# Load data
@time y = loadSingleObj(DATA_PATH)
goodColumns, J = PreProcess.preprocess!(y, maxNanOrNegProp=.9, maxPosProp=.9,
                                        subsample=0.0, rowThresh=-6.0)
Cytof5.Model.logger("good columns: $goodColumns")
y_orig = deepcopy(y)

# replace missing values with `REPLACEMENT`
REPLACEMENT = -6
y = [replaceMissing(yi, REPLACEMENT) for yi in y_orig]

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
                             maxMeta=30,
                             # Seed for reproducible results:
                             seed = 42);

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
source('../FlowSOM/est_Z_from_clusters.R')

idx_upper = cumsum(N)
idx_lower = c(1,idx_upper[-I]+1)
idx = cbind(idx_lower, idx_upper)

fSOMClus = fSOM$meta[fSOM$FlowSOM$map$mapping[,1]]
fsEst = as.list(1:$I)
fsClus = as.numeric(fSOMClus)

mult=1
lines_clus = rep(NA, I)
W = matrix(NA, I, length(unique(fsClus)))
W_relabeled = matrix(NA, I, length(unique(fsClus)))

for (i in 1:$I) {
  clus = fsClus[idx[i,1]:idx[i,2]]
  nclus = length(unique(clus))
  print(nclus) # Number of clusters learned
  W[i, ] = table(clus) / length(clus)
   
  plotPng($RESULTS_DIR %+% 'Y' %+% i %+% '_FlowSOM.png', s=10)
  my.image($(y_orig)[[i]][order(clus),], col=blueToRed(9), zlim=zlim, addL=TRUE,
           na.color='black', cex.y.leg=1, xlab='cell types',  ylab='cells',
           cex.lab=1.5, cex.axis=1.5, xaxt='n',
           f=function(z) {
             addCut(clus, s=10)
             axis(1, at=1:$(J), fg='grey', las=2, cex.axis=1.5)
           })
  dev.off()

  clus = relabel_clusters(clus)
  line_clus = paste(c('i:', i, '| nclu:', nclus), collapse=' ')
  lines_clus[i] = line_clus
  W_relabeled[i, ] = table(clus) / length(clus)
  plotPng($RESULTS_DIR %+% 'Y' %+% i %+% '_FlowSOM_by_w.png', s=10)
  my.image($(y_orig)[[i]][order(clus),], col=blueToRed(9), zlim=zlim, addL=TRUE,
           na.color='black', cex.y.leg=1, xlab='cell types',  ylab='cells',
           cex.lab=1.5, cex.axis=1.5, xaxt='n',
           f=function(z) {
             addCut(clus, s=10)
             axis(1, at=1:$(J), fg='grey', las=2, cex.axis=1.5)
           })
  dev.off()
}

fileConn <- file($RESULTS_DIR %+% "clusters.txt")
writeLines(lines_clus, fileConn)
close(fileConn)
"""

@rget W W_relabeled
for w in (W, W_relabeled)
  fname = (w == W ? "W" : "W_relabeled")

  open("$RESULTS_DIR/$(fname).txt", "w") do file
    for i in 1:size(w, 1)
      wi = join(w[i, :], ",")
      write(file, "$(wi)\n")
    end
  end

  open("$RESULTS_DIR/$(fname).tex", "w") do file
    xtab_W = xtable.xtable(w', digits=3)
    xtab_W = join(split("$xtab_W", "\n")[2:end], "\n")
    write(file, "$(xtab_W)\n")
  end
end


