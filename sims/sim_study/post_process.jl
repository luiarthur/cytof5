using Distributions
using Cytof5, Random, RCall
using JLD2, FileIO

include("util.jl")

#N_factor = parse(Int, ARGS[1]) # 100
#N_factor = 100
#OUTDIR = "result/N$(N_factor)/"
OUTDIR = ARGS[1]
IMGDIR = "$OUTDIR/img/"
run(`mkdir -p $(IMGDIR)`)

println("Loading Data ...")
#@load "$(OUTDIR)/N$(N_factor).jld2" out dat ll lastState c y_dat
@load "$(OUTDIR)/output.jld2" out dat ll lastState c y_dat

I, K = size(dat[:W])
K_MCMC = size(lastState.W, 2)
J = size(lastState.Z, 1)

# Plot loglikelihood
util.plotPdf("$(IMGDIR)/ll.pdf")
util.plot(ll[1000:end], ylab="log-likelihood", xlab="MCMC iteration", typ="l");
util.devOff()

function addGridLines(J::Int, K::Int, col="grey")
  util.abline(v=(1:K) .+ .5, h=(1:J) .+ .5, col=col);
end

# Plot Z
Zpost = util.getPosterior(:Z, out[1])
Zmean = mean(Zpost)

util.plotPdf("$IMGDIR/Z_mean.pdf")
util.myImage(Zmean, xlab="Features", ylab="Markers", addL=true, col=util.greys(11),
             f=Z->addGridLines(J,K_MCMC));
util.devOff()

util.plotPdf("$IMGDIR/Z_mean_est_leftordered.pdf")
util.myImage(Cytof5.Model.leftOrder((Zmean .> .5)*1),
             xlab="Features", ylab="Markers", addL=true, col=util.greys(11),
             f=Z->addGridLines(J,K_MCMC));
util.devOff()

util.plotPdf("$IMGDIR/Z_true.pdf")
util.myImage(dat[:Z], xlab="Features", ylab="Markers");
addGridLines(J, K)
util.devOff()

# Plot W
Wpost = util.getPosterior(:W, out[1])
Wmean = mean(Wpost)

util.plotPdf("$IMGDIR/W_mean.pdf")
util.myImage(Wmean, xlab="Features", ylab="Samples", col=R"greys(10)", addL=true, zlim=[0,.3]);
util.devOff()

util.plotPdf("$IMGDIR/W_true.pdf")
util.myImage(dat[:W], xlab="Features", ylab="Samples", col=R"greys(10)", addL=true, zlim=[0,.3]);
util.devOff()

# Get lam
lamPost = util.getPosterior(:lam, out[1])
unique(lamPost)

# Get b0
b0Post = hcat(util.getPosterior(:b0, out[1])...)'
b0Mean = mean(b0Post, dims=1)
b0Sd = std(b0Post, dims=1)

util.plotPdf("$IMGDIR/b0.pdf")
util.plotPosts(b0Post, cnames=["truth=$b0" for b0 in dat[:b0]]);
util.devOff()

# Get b1
b1Post = hcat(util.getPosterior(:b1, out[1])...)'
b1Mean = mean(b1Post, dims=1)
b1Sd = std(b1Post, dims=1)
util.plotPdf("$IMGDIR/b1.pdf")
util.plotPosts(b1Post, cnames=["truth=$b1" for b1 in dat[:b1]]);
util.devOff()

# Plot Posterior Prob of Missing
util.plotPdf("$IMGDIR/probMissPost.pdf")
R"par(mfrow=c($(I), 1))"
for i in 1:I
  pmiss_mean, pmiss_lower, pmiss_upper, y_seq = util.postProbMiss(b0Post, b1Post, i)
  util.plotPostProbMiss(pmiss_mean, pmiss_lower, pmiss_upper, y_seq, i, main=i)
end
R"par(mfrow=c(1, 1))"
util.devOff()

# Get mus
mus0Post = hcat([m[:mus][0] for m in out[1]]...)'
mus1Post = hcat([m[:mus][1] for m in out[1]]...)'
musPost = [ mus0Post mus1Post ]

util.plotPdf("$IMGDIR/mus.pdf")
R"boxplot"(musPost, ylab="mu*", xlab="", xaxt="n", col="steelblue", pch=20, cex=0);
#util.plot(1:size(musPost, 2), mean(musPost, dims=1), typ="n", ylab="μ*", xlab="", xaxt="n")
#util.addErrbar(R"t(apply($musPost, 2, quantile, c(.025, .975)))", 
#               x=1:size(musPost, 2), ylab="μ*", xlab="", xaxt="n", col="blue", lend=1, lwd=10);
util.abline(h=0, v=size(musPost, 2)/2 + .5, col="grey30", lty=1);
util.abline(h=dat[:mus][0], lty=2, col="steelblue");
util.abline(h=dat[:mus][1], lty=2, col="steelblue");
util.devOff()

# Get sig2
sig2Post = hcat(util.getPosterior(:sig2, out[1])...)'
sig2Mean = mean(sig2Post, dims=1)
sig2Sd = std(sig2Post, dims=1)

util.plotPdf("$IMGDIR/sig2.pdf")
util.plotPosts(sig2Post, cnames=["truth: $s2" for s2 in dat[:sig2]]);
util.devOff()

# Posterior of y_imputed
y_imputed = [ o[:y_imputed] for o in out[2] ]
util.plotPdf("$(IMGDIR)/ydatPost.pdf")
R"par(mfrow=c(4,2))"
for i in 1:I
  for j in 1:J
    util.plot(util.density(dat[:y][i][:, j], na=true), col="red", xlim=[-8,8],
              main="Y sample: $(i), marker: $(j)", bty="n", fg="grey")
    for iter in 1:length(y_imputed)
      yimp = y_imputed[iter]
      util.lines(util.density(yimp[i][:, j]), col=util.rgba("blue", .5))
    end
    util.lines(util.density(dat[:y_complete][i][:, j]), col="grey")
  end
end
R"par(mfrow=c(1,1))"
util.devOff()


idx_missing = [ findall(isnan.(y_dat.y[i])) for i in 1:y_dat.I ]
idx = idx_missing[2][1]
util.plotPdf("$(IMGDIR)/y_trace.pdf")
util.hist([ y_imputed[b][2][idx] for b in 1:length(y_imputed) ], col="blue", border="transparent")
util.plot([ y_imputed[b][2][idx] for b in 1:length(y_imputed) ], typ="l")
util.devOff()

# ARI - adjusted Rand Index ∈  (0, 1). Metric for clustering.
# Higher is better.
open("$IMGDIR/ari.txt", "w") do file
  ariCytof = [ x[1] for x in util.ari.(dat[:lam], lastState.lam) ]
  write(file, "ARI lam: $ariCytof\n")
end

#=
y_141 = [ yimp[1][4, 1] for yimp in y_imputed ]
R"hist"(y_141)
R"plot"(y_141, typ="l")
=#

for i in 1:I
  util.plotPng("$IMGDIR/y_imputed$(i).png")
  util.yZ_inspect(out[1], i=i, lastState.y_imputed, zlim=[-4,4], using_zero_index=false) 
  util.devOff()

  util.plotPng("$IMGDIR/y_true$(i).png")
  util.yZ_inspect(out[1], i=i, dat[:y], zlim=[-4,4], na="black", using_zero_index=false)
  util.devOff()
end

open("$IMGDIR/priorBeta.txt", "w") do file
  b0Prior = join(c.b0_prior, "\n")
  b1Prior = join(c.b1_prior, "\n")
  write(file, "b0Prior:\n$b0Prior\nb1Prior:\n$b1Prior\n")
end

