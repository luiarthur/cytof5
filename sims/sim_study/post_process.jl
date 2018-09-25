import Pkg
Pkg.activate("../../")

using Distributions
using Cytof5, Random, RCall
using JLD2, FileIO

include("util.jl")

#N_factor = parse(Int, ARGS[1]) # 100
N_factor = 100
OUTDIR = "result/N$(N_factor)/"

println("Loading Data ...")
@load "$(OUTDIR)/N$(N_factor).jld2" out dat ll lastState

I, K = size(dat[:W])
K_MCMC = size(lastState.W, 2)
J = size(lastState.Z, 1)

# Load cytof3, rcommon library in R
R"library(cytof3)"
R"library(rcommon)"

# Import R plotting functions
plot = R"plot";
plotPost = R"rcommon::plotPost"
plotPosts = R"rcommon::plotPosts"
myImage = R"cytof3::my.image"
plotPdf = R"pdf"
devOff = R"dev.off"
blueToRed = R"blueToRed"
greys = R"cytof3::greys"
plot_dat = R"cytof3::plot_dat"

# Plot loglikelihood
plot(ll[100:end], ylab="log-likelihood", xlab="MCMC iteration", typ="l");

# Plot Z
Zpost = util.getPosterior(:Z, out[1])
Zmean = util.matMean(Zpost)
myImage(Zmean, xlab="Features", ylab="Markers", addL=true, col=greys(11));

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

# Plot Data Hist 
run(`mkdir -p $(OUTDIR)/img`)
plotPdf("$(OUTDIR)/img/ydatPost.pdf")
R"par(mfrow=c(4,2))"
for i in 1:I
  for j in 1:J
    plot_dat(lastState.y_imputed, i, j, xlim=[-8,8])
  end
end
R"par(mfrow=c(1,1))"
devOff()
