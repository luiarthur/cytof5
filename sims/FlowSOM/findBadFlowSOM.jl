using Random
using RCall
using Cytof5

# Import R libs
R"""
library(xtable)
library(FlowSOM)
library(flowCore)
library(rcommon)
library(cytof3)
source("est_Z_from_clusters.R")
set.seed(3)
"""

OUTPUT_DIR = "results/flowSearch/"
mkpath(OUTPUT_DIR)

# Include utils 
include("../sim_study/util.jl")


function replaceMissing(yi, x)
  out = deepcopy(yi)
  out[isnan.(out)] .= x
  return out
end


function sim(jl_seed::Int)
  Random.seed!(jl_seed)
  mkpath("$OUTPUT_DIR/$jl_seed/")
  I=3
  J=32
  N = [3, 1, 2] * 100
  K=8
  L=5
  Z=Cytof5.Model.genZ(J, K, .6)
  dat = Cytof5.Model.genData(I, J, N, K, L, Z,
                             Dict(:b0=>-9.2, :b1=>2.3), # missMechParams
                             [0.2, 0.1, 0.3], # sig2
                             Dict(0=>-(0.5 .+ rand(L) * 4.5), #mus
                                  1=>  0.5 .+ rand(L) * 4.5),
                             rand(K)*10, # a_W
                             Dict([ z => rand(L)*10 for z in 0:1 ]), # a_eta
                             sortLambda=false, propMissingScale=0.7)
  y_dat = Cytof5.Model.Data(dat[:y])

  util.plotPdf("$OUTPUT_DIR/$jl_seed/Z.pdf")
  util.myImage(dat[:Z])
  util.devOff()

  open("$OUTPUT_DIR/$jl_seed/dat.txt", "w") do f
    write(f, "mu*0: $(dat[:mus][0]) \n")
    write(f, "mu*1: $(dat[:mus][1]) \n")
    write(f, "sig2: $(dat[:sig2]) \n")
  end

  # Preimpute missing values
  y_tilde = [ replaceMissing(yi, -20) .+ 0 for yi in y_dat.y ]
  Y_tilde = R"Reduce(rbind, $y_tilde)"
  @rput Y_tilde
  ff_Y = R"colnames(Y_tilde) <- 1:NCOL(Y_tilde); flowFrame(Y_tilde)"

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
  png($OUTPUT_DIR %+% '/' %+% $jl_seed %+% '/'%+% 'YZ%03d_FlowSOM.png', height=500*mult, width=500*mult)
  for (i in 1:$I) {
    clus = fsClus[idx[i,1]:idx[i,2]]
    print(length(unique(clus))) # Number of clusters learned
    clus = relabel_clusters(clus)
    my.image($(y_dat.y)[[i]][order(clus),], col=blueToRed(11), zlim=c(-4,4), addL=TRUE,
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
  
  sink($OUTPUT_DIR %+% '/' %+% $jl_seed %+% '/' %+% "ari.txt")
    println("ARI:")
    print(ARI)
    println("ARI all:")
    print(ARI_all)
  sink()
  """
end


SIMS = 100
Threads.@threads for i in 1:SIMS
  println("$i / $SIMS")
  sim(i)
end
