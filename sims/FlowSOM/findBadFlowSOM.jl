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
runif(n; a=0, b=1) = rand(n) * (b - a) .+ a
csrand(n; min=0) = cumsum(rand(n) * 2) .+ min
flipbit(x; prob=.5) = [prob > rand() ? 1 - xi : xi for xi in x]

function sim(jl_seed::Int, n_fac::Int; K=5, L=Dict(0=>3, 1=>3), J=20, fs_seed=42, save=false,
             results_dir="results/flowSearch/", eps=zeros(3))
  OUTPUT_DIR = "$(results_dir)/N$(n_fac)/K$(K)/$jl_seed"
  mkpath(OUTPUT_DIR)

  N = [8, 1, 2] * n_fac
  I = length(N)
  Random.seed!(jl_seed)
  Z = zeros(Int, J, K)
  while !Cytof5.Model.isValidZ(Z)
    Z=Cytof5.Model.genZ(J, K, .5)
    Z[:, 2] .= flipbit(Z[:, 1], prob=.1)
  end
  # mus=Dict(0 => -csrand(L[0], min=.5),
  #          1 =>  csrand(L[1], min=.5))
  # mus=Dict(0 => -crange(1, 4, L[0]) .+ randn(L[0]) * .2,
  #          1 =>  crange(1, 3, L[1]) .+ randn(L[1]) * .2)
  # mus=Dict(0 => -[.5, 1.5, 3.0], 
  #          1 => +[.5, 1.5, 2.7])
  mus=Dict(0 => -[1.0, 2.3, 3.5], 
           1 => +[1.0, 2.0, 3.0])
  a_W=rand(K)*10
  a_eta=Dict(z => rand(L[z])*10 for z in 0:1)
  simdat = Cytof5.Model.genData(J=J, N=N, K=K, L=L, Z=Z,
                                beta=[-9.2, -2.3],
                                sig2=[0.2, 0.1, 0.3],
                                mus=mus,
                                a_W=a_W,
                                a_eta=a_eta,
                                sortLambda=false, propMissingScale=0.7, eps=eps)
  dat = Cytof5.Model.Data(simdat[:y])

  util.plotPdf("$OUTPUT_DIR/Z.pdf")
  util.myImage(simdat[:Z])
  util.devOff()

  open("$OUTPUT_DIR/dat.txt", "w") do f
    write(f, "mu*0: $(simdat[:mus][0]) \n")
    write(f, "mu*1: $(simdat[:mus][1]) \n")
    write(f, "sig2: $(simdat[:sig2]) \n")
    write(f, "eps: $(simdat[:eps]) \n")

    for i in 1:I
      write(f, "W$(i): $(simdat[:W][i, :]) \n")
    end
    write(f, "\n")
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
                       # nClus=K,
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
  plotPng($OUTPUT_DIR %+% '/' %+% 'YZ%03d_FlowSOM.png', s=mult)
  for (i in 1:$I) {
    clus = fsClus[idx[i,1]:idx[i,2]]
    print('fs num clus (i' %+% i %+% '): ' %+% length(unique(clus))) # Number of clusters learned
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

  sink($OUTPUT_DIR %+% '/' %+% "ari.txt")
    println("ARI:")
    print(ARI)
    println("ARI all:")
    print(ARI_all)
    println('')
  sink()
  """

  # if any(R"ARI" .< .5)
  #   println("saving sim $jl_seed")
  #   BSON.@save "$(OUTPUT_DIR)/$(jl_seed)/simdat.bson" simdat
  # end

  if save
    BSON.@save "$(OUTPUT_DIR)/simdat.bson" simdat
  end
end


# MAIN
# SIMS = [1, 90, 98, 68] # 90, 98 are good ones
N_FAC = [500, 5000]
# K_DICT = Dict(N_FAC[1] => 5, N_FAC[2]=> 10)
# SIMS = Dict(N_FAC[1] => [90], N_FAC[2] => [98])
# SIMS = Dict(N_FAC[1] => collect(90:100), N_FAC[2] => collect(90:100))

# K_DICT = Dict(N_FAC[1] => [5, 8], N_FAC[2]=> [5, 10, 15])
# SIMS = Dict(N_FAC[1] => [90, 98, 1], N_FAC[2] => [90, 98, 1])

K_DICT = Dict(N_FAC[1] => [5], N_FAC[2]=> [10])
SIMS = Dict(N_FAC[1] => [90], N_FAC[2] => [1])

@assert length(K_DICT) == length(SIMS) == length(N_FAC)

# These don't have much effect. Bad is bad.
FS_SEED = 42 # PASS
# FS_SEED = 0 # PASS
# FS_SEED = 1 # PASS

#Threads.@threads for i in 1:SIMS
for n_fac in N_FAC
  for jl_seed in SIMS[n_fac]
    for k in K_DICT[n_fac]
      println("$(jl_seed) | n_fac: $(n_fac) | K_TRUE: $(k)")
      sim(jl_seed, n_fac, K=k, L=Dict(0=>3, 1=>3), J=20, fs_seed=FS_SEED, save=true,
          results_dir="data/kills-flowsom/", eps=fill(.005, 3))
    end
  end
end
