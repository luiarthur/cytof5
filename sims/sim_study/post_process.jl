import Pkg
Pkg.activate("../../")

using Distributions
using Cytof5, Random, RCall
using JLD2, FileIO

include("util.jl")

#N_factor = parse(Int, ARGS[1]) # 100
N_factor = 1000
OUTDIR = "result/N$(N_factor)/"

println("Loading Data ...")
@load "$(OUTDIR)/N$(N_factor).jld2" out dat ll lastState

I, K = size(dat[:W])
K_MCMC = size(lastState.W, 2)
J = size(lastState.Z, 1)

# Load cytof3, rcommon library in R

# Import R plotting functions
plot = R"plot";
ari = R"cytof3::ari";
rgba = R"cytof3::rgba";
density = R"density";
lines = R"lines";
plotPost = R"rcommon::plotPost"
plotPosts = R"rcommon::plotPosts"
myImage = R"cytof3::my.image"
plotPdf = R"pdf"
devOff = R"dev.off"
blueToRed = R"cytof3::blueToRed"
greys = R"cytof3::greys"
plot_dat = R"cytof3::plot_dat"
yZ_inspect = R"cytof3::yZ_inspect"

# Plot loglikelihood
plot(ll[100:end], ylab="log-likelihood", xlab="MCMC iteration", typ="l");

function addGridLines(J::Int, K::Int, col="grey")
  R"abline"(v=(1:K) .+ .5, h=(1:J) .+ .5, col=col);
end

# Plot Z
Zpost = util.getPosterior(:Z, out[1])
Zmean = util.matMean(Zpost)
myImage(Zmean, xlab="Features", ylab="Markers", addL=true, col=greys(11), f=Z->addGridLines(J,K));

myImage(dat[:Z], xlab="Features", ylab="Markers");
addGridLines(J, K)

# Plot W
Wpost = util.getPosterior(:W, out[1])
Wmean = util.matMean(Wpost)
myImage(Wmean, xlab="Features", ylab="Samples", col=R"greys(10)", addL=true,
        zlim=[0,.3]);

# Get lam
lamPost = util.getPosterior(:lam, out[1])
unique(lamPost)

# Get b0
b0Post = hcat(util.getPosterior(:b0, out[1])...)'
b0Mean = mean(b0Post, dims=1)
b0Sd = std(b0Post, dims=1)
plotPosts(b0Post);

# Get b1
b1Post = hcat(util.getPosterior(:b1, out[1])...)'
b1Mean = mean(b1Post, dims=1)
b1Sd = std(b1Post, dims=1)
plotPosts(b1Post);

# Plot Posterior Prob of Missing
include("util.jl")
R"par(mfrow=c($(I), 1))"
for i in 1:I
  pmiss_mean, pmiss_lower, pmiss_upper, y_seq = util.postProbMiss(b0Post, b1Post, i)
  util.plotPostProbMiss(pmiss_mean, pmiss_lower, pmiss_upper, y_seq, i, main=i)
end
R"par(mfrow=c(1, 1))"


# Get sig2
sig2Post = hcat(util.getPosterior(:sig2, out[1])...)'
sig2Mean = mean(sig2Post, dims=1)
sig2Sd = std(sig2Post, dims=1)
plotPosts(sig2Post);

# Plot y_imputed
lam1Sortperm = sortperm(lastState.lam[1])
myImage(lastState.y_imputed[1], col=blueToRed(7), zlim=[-4,4], addL=true,
        xlab="markers", ylab="obs", na="black", main="Imputed y[1] (last sample in MCMC)");
myImage(lastState.y_imputed[1][lam1Sortperm, :],
        col=blueToRed(7), zlim=[-4,4], addL=true,
        xlab="markers", ylab="obs", na="black", main="Sorted Imputed y[1] (last sample in MCMC)");

myImage(dat[:y][1], col=blueToRed(7), zlim=[-4,4], addL=true,
        xlab="markers", ylab="obs", na="black", main="Data: y[1]");
myImage(dat[:y][1][lam1Sortperm, :], col=blueToRed(7), zlim=[-4,4], addL=true,
        xlab="markers", ylab="obs", na="black", main="Data: y[1]");

# Posterior of y_imputed
run(`mkdir -p $(OUTDIR)/img`)
y_imputed = [ o[:y_imputed] for o in out[2] ]
plotPdf("$(OUTDIR)/img/ydatPost.pdf")
R"par(mfrow=c(4,2))"
for i in 1:I
  for j in 1:J
    for iter in 1:length(y_imputed)
      yimp = y_imputed[iter]
      if iter == 1
        plot(density(yimp[i][:, j]), col=rgba("blue", .5), xlim=[-8,8],
             main="Y sample: $(i), marker: $(j)", bty="n", fg="grey")
      else
        lines(density(yimp[i][:, j]), col=rgba("blue", .5))
      end
      lines(density(dat[:y_complete][i][:, j]), col="grey")
    end
    # Plot simulated data
  end
end
R"par(mfrow=c(1,1))"
devOff()


# ARI - adjusted Rand Index ∈  (0, 1). Metric for clustering.
# Higher is better.
ariCytof = [ x[1] for x in ari.(dat[:lam], lastState.lam) ]

#=
y_141 = [ yimp[1][4, 1] for yimp in y_imputed ]
R"hist"(y_141)
R"plot"(y_141, typ="l")
=#

yZ_inspect(out[1], i=3, lastState.y_imputed, zlim=[-8,8], using_zero_index=false) 
yZ_inspect(out[1], i=3, dat[:y], zlim=[-8,8], na="black", using_zero_index=false)
