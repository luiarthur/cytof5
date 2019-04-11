using Random
using RCall
using Cytof5
using BSON

# Import R libs
R"""
library(FlowSOM)
library(flowCore)
library(rcommon)
library(cytof3)
source("est_Z_from_clusters.R")
set.seed(3)
"""

# Include utils 
include("../sim_study/util.jl")

function replaceMissing(yi, x)
  out = deepcopy(yi)
  out[isnan.(out)] .= x
  return out
end

crange(start, stop, length) = collect(range(start, stop=stop, length=length))

function sim(jl_seed::Int, n_fac::Int; K=5, L=Dict(0=>3, 1=>3), J=20, fs_seed=42)
  OUTPUT_DIR = "results/flowSearch/N$(n_fac)/"
  mkpath(OUTPUT_DIR)

  mkpath("$OUTPUT_DIR/$jl_seed/")

  N = [8, 1, 2] * n_fac
  I = length(N)
  Random.seed!(jl_seed)
  Z=Cytof5.Model.genZ(J, K, .5)
  simdat = Cytof5.Model.genData(J=J, N=N, K=K, L=L, Z=Z,
                             beta=[-9.2, -2.3],
                             sig2=[0.2, 0.1, 0.3],
                             mus=Dict(0 => -crange(1, 4, L[0]) .+ randn(L[0]) * .1,
                                      1 =>  crange(1, 3, L[1]) .+ randn(L[1]) * .1),
                             a_W=rand(K)*10,
                             a_eta=Dict(z => rand(L[z])*10 for z in 0:1),
                             sortLambda=false, propMissingScale=0.7)
  dat = Cytof5.Model.Data(simdat[:y])

  util.plotPdf("$OUTPUT_DIR/$jl_seed/Z.pdf")
  util.myImage(simdat[:Z])
  util.devOff()

  open("$OUTPUT_DIR/$jl_seed/dat.txt", "w") do f
    write(f, "mu*0: $(simdat[:mus][0]) \n")
    write(f, "mu*1: $(simdat[:mus][1]) \n")
    write(f, "sig2: $(simdat[:sig2]) \n")
  end

  # Preimpute missing values
  REPLACEMENT = -6.0
  y = [replaceMissing(yi, REPLACEMENT) for yi in dat.y]

  Y = vcat(y...)
  @rput Y
  ff_Y = R"colnames(Y) <- 1:NCOL(Y); flowFrame(Y)"

  FlowSOM = R"FlowSOM::FlowSOM"
  @time fSOM = FlowSOM(ff_Y,
                       # Input options:
                       colsToUse = 1:J,
                       # Metaclustering options:
                       #nClus = 20,
                       maxMeta=20,
                       # Seed for reproducible results:
                       seed=fs_seed)

  @rput fSOM I J N

  R"""
  zlim = c(-1, 1) * 4
  plotPng = function(fname, s=10, w=480, h=480, ps=12, ...) {
    png(fname, w=w*s, h=h*s, pointsize=ps*s, ...)
  }
  addCut = function(clus, s=1) abline(h=cumsum(table(clus)) + .5, lwd=3*s, col='yellow')
  addGridLines = function(Z, s=1) abline(v=1:NCOL(Z) + .5, h=1:NROW(Z) + .5, col='grey', lwd=1*s)

  idx_upper = cumsum(N)
  idx_lower = c(1,idx_upper[-I]+1)
  idx = cbind(idx_lower, idx_upper)
  
  fSOMClus = fSOM$meta[fSOM$FlowSOM$map$mapping[,1]]
  fsEst = as.list(1:$I)
  fsClus = as.numeric(fSOMClus)
  
  mult=10
  plotPng($OUTPUT_DIR %+% '/' %+% $jl_seed %+% '/'%+% 'YZ%03d_FlowSOM.png', s=mult)
  for (i in 1:$I) {
    clus = fsClus[idx[i,1]:idx[i,2]]
    print(length(unique(clus))) # Number of clusters learned
    clus = relabel_clusters(clus)
    my.image($(dat.y)[[i]][order(clus),], col=blueToRed(9), zlim=zlim, addL=TRUE,
             na.color='black', cex.y.leg=1, xlab='cell types',  ylab='cells',
             cex.lab=1.5, cex.axis=1.5, xaxt='n',
             f=function(z) {
               addCut(clus, s=mult)
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
  
  println("ARI:" %+% ARI)
  println("ARI all:" %+% ARI_all)

  sink($OUTPUT_DIR %+% '/' %+% $jl_seed %+% '/' %+% "ari.txt")
    println("ARI:")
    print(ARI)
    println("ARI all:")
    print(ARI_all)
  sink()
  """

  if any(R"ARI" .< .5)
    println("saving sim $jl_seed")
    BSON.@save "$(OUTPUT_DIR)/$(jl_seed)/simdat.bson" simdat
  end
end


# MAIN
SIMS = [1, 90, 98, 68] # 90, 98 are good ones
K_DICT = Dict(500 => 5, 5000 => 10)
N_FAC = sort(collect(keys(K_DICT)))

# These don't have much effect. Bad is bad.
# FS_SEED = 42
# FS_SEED = 0 # KEEP
# FS_SEED = 1 # KEEP
FS_SEED = 0 # KEEP

#Threads.@threads for i in 1:SIMS
for n_fac in N_FAC
  for jl_seed in SIMS
    println("$(jl_seed) | n_fac: $(n_fac)")
    sim(jl_seed, n_fac, K=K_DICT[n_fac], L=Dict(0=>3, 1=>3), J=20, fs_seed=FS_SEED)
  end
end

# NOTE:
# use SIM 90 for N_factor 500
#     SO< 98 for N_factor 5000
