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

# Plot loglikelihood
plot(ll[100:end], ylab="log-likelihood", xlab="MCMC iteration", typ="l");

# Plot Z
Zpost = util.getPosterior(:Z, out[1])
Zmean = util.matMean(Zpost)
myImage(Zmean, xlab="Features", ylab="Markers");

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
I = size(b0Post, 2)
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

# Plot y_imputed# 
myImage(lastState.y_imputed[1], col=blueToRed(7), zlim=[-4,4], addL=true,
        xlab="markers", ylab="obs", na="black");
myImage(dat[:y][1], col=blueToRed(7), zlim=[-4,4], addL=true,
        xlab="markers", ylab="obs", na="black");

