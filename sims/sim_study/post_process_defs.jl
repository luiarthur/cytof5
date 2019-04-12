# Required for BSON loading output.bson
using Cytof5, Distributions

using Random

# TODO: Remove this dependency
using RCall

# TODO: remove these in favor of BSON
# using JLD2, FileIO
using BSON

include("util.jl")
#include("CytofImg.jl")

"""
# Example

```julia
post_process("path/to/output.bson")
```
"""
function post_process(PATH_TO_OUTPUT, thresh=0.9, min_presences=[0, .01, .03, .05]) # bla/output.bson
  OUTDIR = join(split(PATH_TO_OUTPUT, "/")[1:end-1], "/")
  IMGDIR = "$OUTDIR/img/"
  mkpath(IMGDIR)

  println("Loading Data ...")
  BSON.@load "$PATH_TO_OUTPUT" out ll lastState c metrics init dden simdat
  y_dat = simdat[:y]

  I, K = size(simdat[:W])
  K_MCMC = size(lastState.W, 2)
  J = size(lastState.Z, 1)
  MCMC_ITER = length(out[1])
  BURN = length(ll) - MCMC_ITER

  # Plot loglikelihood
  util.plotPdf("$(IMGDIR)/ll.pdf")
  util.plot(ll[(end-MCMC_ITER):end], ylab="log-likelihood", xlab="MCMC iteration (post-burn)",
            typ="l");
  util.devOff()

  util.plotPdf("$(IMGDIR)/ll_entire_history.pdf")
  util.plot(ll, ylab="log-likelihood", xlab="MCMC iteration", typ="l");
  util.abline(v=BURN);
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
  util.myImage(simdat[:Z], xlab="Features", ylab="Markers");
  addGridLines(J, K)
  util.devOff()

  # Plot W
  Wpost = util.getPosterior(:W, out[1])
  Wpost = cat(Wpost..., dims=3) # I x K_MCMC x NMCMC

  util.plotPdf("$IMGDIR/W.pdf")
  for i in 1:I
    util.boxplot(Wpost[i, :, :]', xlab="Features", ylab="W$(i)", );
    # Plot data
    util.abline(h=simdat[:W][i, :], lty=2, col="grey")
  end
  util.devOff()

  # Plot alpha
  alphaPost = util.getPosterior(:alpha, out[1])
  println("Making alpha ...")
  util.plotPdf("$(IMGDIR)/alpha.pdf")
  util.plotPost(alphaPost, ylab="density", xlab="alpha", main="");
  util.devOff()

  # Get lam
  lamPost = util.getPosterior(:lam, out[1])
  # unique(lamPost)

  # Get mus
  delta0Post = cat([m[:delta][0] for m in out[1]]..., dims=2) # L0 x NMCMC
  delta1Post = cat([m[:delta][1] for m in out[1]]..., dims=2) # L1 x NMCMC
  mus0Post = -cumsum(delta0Post, dims=1)
  mus1Post = cumsum(delta1Post, dims=1)
  musPost = [ mus0Post; mus1Post ]

  util.plotPdf("$IMGDIR/mus.pdf")
  R"boxplot"(musPost', ylab="mu*", xlab="", xaxt="n", col="steelblue", pch=20, cex=0);
  #util.plot(1:size(musPost, 2), mean(musPost, dims=1), typ="n", ylab="μ*", xlab="", xaxt="n")
  #util.addErrbar(R"t(apply($musPost, 2, quantile, c(.025, .975)))", 
  #               x=1:size(musPost, 2), ylab="μ*", xlab="", xaxt="n", col="blue", lend=1, lwd=10);
  util.abline(h=0, v=size(mus0Post, 2) + .5, col="grey30", lty=1);
  util.abline(h=simdat[:mus][0], lty=2, col="steelblue");
  util.abline(h=simdat[:mus][1], lty=2, col="steelblue");
  util.devOff()

  # Get sig2
  sig2Post = hcat(util.getPosterior(:sig2, out[1])...) # I x NMCMC

  util.plotPdf("$IMGDIR/sig2.pdf")
  util.boxplot(sig2Post', xlab="samples", ylab="sig2")
  util.abline(h=simdat[:sig2], lty=2, col="grey")
  util.devOff()

  # Posterior of y_imputed
  y_imputed = [ o[:y_imputed] for o in out[2] ]
  util.plotPdf("$(IMGDIR)/ydatPost.pdf")
  R"par(mfrow=c(4,2))"
  for i in 1:I
    for j in 1:J
      util.plot(util.density(simdat[:y][i][:, j], na=true), col="red", xlim=[-8,8],
                main="Y sample: $(i), marker: $(j)", bty="n", fg="grey")
      for iter in 1:length(y_imputed)
        yimp = y_imputed[iter]
        util.lines(util.density(yimp[i][:, j]), col=util.rgba("blue", .5))
      end
      util.lines(util.density(simdat[:y_complete][i][:, j]), col="grey")
    end
  end
  R"par(mfrow=c(1,1))"
  util.devOff()


  idx_missing = [ findall(isnan.(y_dat[i])) for i in 1:I ]
  idx = idx_missing[2][1]
  util.plotPdf("$(IMGDIR)/y_trace.pdf")
  util.hist([ y_imputed[b][2][idx] for b in 1:length(y_imputed) ], col="blue", border="transparent")
  util.plot([ y_imputed[b][2][idx] for b in 1:length(y_imputed) ], typ="l")
  util.devOff()

  # ARI - adjusted Rand Index ∈  (0, 1). Metric for clustering.
  # Higher is better.
  open("$IMGDIR/ari.txt", "w") do file
    ariCytof = [ x[1] for x in util.ari.(simdat[:lam], lastState.lam) ]
    write(file, "ARI lam: $ariCytof\n")
  end

  println("Make png...")

  # Set png resolution settings
  s_png = 10
  fy(clus) = util.addCut(clus, s_png)
  fZ(Z) = util.addGridLines(Z, s_png)

  for i in 1:I
    util.plotPng("$IMGDIR/y_imputed$(i).png")
    util.yZ_inspect(out[1], i=i, lastState.y_imputed, zlim=[-4,4], using_zero_index=false,
                    thresh=thresh, fy=fy, fZ=fZ)
    util.devOff()

    util.plotPng("$IMGDIR/y_dat$(i).png")
    util.yZ_inspect(out[1], i=i, simdat[:y], zlim=[-4,4], na="black", using_zero_index=false,
                    thresh=thresh, fy=fy, fZ=fZ)
    util.devOff()
  end
  println("Done with png...")

  #= Plots.jl
  for i in 1:I
    CytofImg.yZ_inspect(out[1], lastState.y_imputed, i, thresh=.9)
    CytofImg.Plots.savefig("$IMGDIR/y_imputed$(i).png")

    CytofImg.yZ_inspect(out[1], simdat[:y], i, thresh=.9)
    CytofImg.Plots.savefig("$IMGDIR/y_dat$(i).png")
  end
  =#

  # Missing Mechanism
  open("$IMGDIR/beta.txt", "w") do file
    for i in 1:I
      bi = join(c.beta[:, i], ", ")
      write(file, "for i=$i, beta = $bi \n")
    end
  end

  # Plot missing mechanism
  util.plotPdf("$IMGDIR/prob_miss.pdf")
  R"par(mfrow=c($(I), 1))"
  for i in 1:I
    util.plotProbMiss(c.beta, i)
  end
  R"par(mfrow=c(1,1))"
  util.devOff()

  # TODO: DO THIS
  # Plot QQ 
  # util.plotPdf("$IMGDIR/qq.pdf")
  # R"par(mfrow=c(3, 3), mar=c(5.1, 4, 2, 1))"
  # y_obs_range = util.y_obs_range(y_dat)
  # for i in 1:I
  #   for j in 1:J
  #     print("\ri: $i, j: $j")
  #     # QQ of observed expression levels
  #     y_obs, y_pp = util.qq_yobs_postpred(y_dat, i, j, out)
  #     util.myQQ(y_obs, y_pp, pch=20, ylab="post pred quantiles",
  #               xlab="y (observed) quantiles", main="i: $i, j: $j", xlim=y_obs_range,
  #               ylim=y_obs_range)
  #   end
  # end
  # R"par(mfrow=c(1, 1), mar=mar.default())"
  # println()
  # util.devOff()

  # TODO: consider removing this
  # # QQ with observed values
  # util.plotPdf("$IMGDIR/qq_truedat.pdf")
  # R"par(mfrow=c(3, 3), mar=c(5.1, 4, 2, 1))"
  # y_obs_range = util.y_obs_range(simdat[:y_complete])
  # for i in 1:I
  #   for j in 1:J
  #     println("i: $i, j: $j")
  #     # QQ of observed expression levels
  #     y_obs, y_pp = util.qq_yobs_postpred(simdat[:y_complete], i, j, out)
  #     util.myQQ(y_obs, y_pp, pch=20, ylab="post pred quantiles", xlab="y
  #               (observed) quantiles", main="i: $i, j: $j", xlim=y_obs_range,
  #               ylim=y_obs_range)
  #   end
  # end
  # R"par(mfrow=c(1, 1), mar=mar.default())"
  # util.devOff()


  # Plot Posterior Density for i, j observed
  mkpath("$(IMGDIR)/dden")
  for i in 1:I
    for j in 1:J
      println("Plot posterior density for (i: $i, j: $j) observed ...")
      util.plotPdf("$(IMGDIR)/dden/dden_i$(i)_j$(j).pdf")
      dyij = util.density(filter(yij -> !isnan(yij), y_dat[i][:, j]))
      dd_ij = hcat([dd[i, j] for dd in dden]...)
      pdyij_mean = R"rowMeans($dd_ij)" .+ 0
      pdyij_lower = R"apply($dd_ij, 1, quantile, .025)" .+ 0
      pdyij_upper = R"apply($dd_ij, 1, quantile, .975)" .+ 0
      h = maximum([dyij[:y] .+ 0; pdyij_upper])
      util.plot(c.y_grid, pdyij_mean, xlab="y", ylab="density", ylim=[0, h],
                main="i: $i, j: $j", col="blue", lwd=2, typ="l")
      util.colorBtwn(c.y_grid, pdyij_lower, pdyij_upper, from=-10, to=10,
                     col=util.rgba("blue", .3))
      util.lines(dyij, col="grey", lwd=2)
      util.devOff()
    end
  end


  # Separate graphs
  mkpath("$IMGDIR/sep")
  for i in 1:I
    println("Separate graphs for yZ $(i)...")
    idx_best = R"estimate_ZWi_index($(out[1]), $i)"[1]
    Zi = out[1][idx_best][:Z]
    Wi = out[1][idx_best][:W][i,:]

    # Point Est for Wi and Zi. txt and pdf.
    open("$IMGDIR/sep/W_$(i)_hat.txt", "w") do file
      write(file, "$(join(Wi, "\n"))\n")
    end

    open("$IMGDIR/sep/W$(i)_hat_ordered_cumsum.txt", "w") do file
      ord = sortperm(Wi, rev=true)
      cs_wi_sorted = cumsum(Wi[ord])
      write(file, "num_features,k,wi,cumprop\n")
      for k in 1:K_MCMC
        write(file, "$(k),$(ord[k]),$(Wi[ord][k]),$(cs_wi_sorted[k])\n")
      end
    end

    open("$IMGDIR/sep/Z$(i)_hat.txt", "w") do file
      for j in 1:J
        zj = join(Int.(Zi[j, :]), ",")
        write(file, "$(zj)\n")
      end
    end

    lami = out[1][idx_best][:lam][i]
    ord = sortperm(Wi, rev=true)
    lami = util.reorder_lami(ord, lami)

    for min_presence in min_presences
      common_celltypes = util.get_common_celltypes(Wi, thresh=min_presence,
                                                   filter_by_min_presence=true)
      println("common celltypes (min_presence > $min_presence): $common_celltypes")
      K_trunc = length(common_celltypes)

      util.plotPng("$IMGDIR/sep/y_dat$(i)_only_minpresence$(min_presence).png")
      ord_yi = sortperm(lami)
      util.myImage(y_dat[i][ord_yi[1 .<= lami[ord_yi] .<= K_trunc], :],
                   addL=true, f=yi->util.addCut(lami, s_png),
                   zlim=[-4,4], col=util.blueToRed(9), na="black", xlab="markers",
                   ylab="cells");
      util.devOff()

      util.plotPdf("$IMGDIR/sep/Z_hat$(i)_minpresence$(min_presence).pdf", w=5, h=10)
      util.myImage(Zi[:, common_celltypes], addL=false, ylab="markers", yaxt="n",
                   f=Z->addGridLines(J, K_trunc), xaxt="n", xlab="celltypes");

      perc = string.(round.(Wi[common_celltypes] * 100, digits=2), "%")
      R"""
      axis(3, at=1:$K_trunc, label=$(perc), las=2, fg="grey", cex.axis=1)
      axis(1, at=1:$K_trunc, label=$(common_celltypes), las=1,
           fg="grey", cex.axis=1)
      axis(2, at=1:$J, label=1:$J, las=2, fg="grey", cex.axis=1)
      """
      util.devOff()
    end
  end # separate graphs

end
