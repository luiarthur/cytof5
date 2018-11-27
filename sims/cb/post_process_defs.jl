using Distributions
using Cytof5, Random, RCall
using JLD2, FileIO
include("../sim_study/util.jl")

function loadSingleObj(objPath)
  data = load(objPath)
  return data[collect(keys(data))[1]]
end


function post_process(path_to_output)
  jld_filename = split(path_to_output, "/")[end]
  outdir = join(split(path_to_output, "/")[1:end-1], "/")
  # datapath = "data/reduced_cb.jld2"
  datapath = "$outdir/reduced_data/reduced_cb.jld2"
  println("datapath: $datapath")
  IMGDIR = "$outdir/img/"

  run(`mkdir -p $(IMGDIR)`)

  println("Loading Data ...")
  cbData = loadSingleObj(datapath)
  println("Loading Results ...")
  @load path_to_output out ll lastState c

  I, K_MCMC = size(lastState.W)
  J = size(lastState.Z, 1)

  # Plot loglikelihood
  util.plotPdf("$(IMGDIR)/ll_complete_history.pdf")
  util.plot(ll, ylab="log-likelihood", xlab="MCMC iteration", typ="l");
  util.abline(v=length(ll) - length(out[1]))
  util.devOff()

  util.plotPdf("$(IMGDIR)/ll_postburn.pdf")
  util.plot(ll[end-length(out[1]):end], ylab="log-likelihood", xlab="MCMC iteration", typ="l");
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

  # Plot W
  Wpost = util.getPosterior(:W, out[1])
  Wmean = mean(Wpost)

  util.plotPdf("$IMGDIR/W_mean.pdf")
  util.myImage(Wmean, xlab="Features", ylab="Samples", col=R"greys(10)", addL=true, zlim=[0,.3]);
  util.devOff()

  # Get lam
  lamPost = util.getPosterior(:lam, out[1])
  unique(lamPost)

  # Missing Mechanism
  open("$IMGDIR/beta.txt", "w") do file
    for i in 1:I
      bi = join(c.beta[:, i], ", ")
      write(file, "for i=$i, beta = $bi \n")
    end
  end

  # Plot missing mechanism
  util.plotPdf("$IMGDIR/prob_miss.pdf")
  R"par(mfrow=c($I, 1))"
  for i in 1:I
    util.plotProbMiss(c.beta, i)
  end
  R"par(mfrow=c(1,1))"
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
  util.abline(h=0, v=size(mus0Post, 2) + .5, col="grey30", lty=1);
  util.devOff()

  # Get sig2
  sig2Post = hcat(util.getPosterior(:sig2, out[1])...)'
  sig2Mean = mean(sig2Post, dims=1)
  sig2Sd = std(sig2Post, dims=1)

  util.plotPdf("$IMGDIR/sig2.pdf")
  util.plotPosts(sig2Post);
  util.devOff()

  # Posterior of y_imputed
  y_imputed = [ o[:y_imputed] for o in out[2] ]
  util.plotPdf("$(IMGDIR)/ydatPost.pdf")
  R"par(mfrow=c(4,2))"
  for i in 1:I
    for j in 1:J
      println("i: $i, j: $j")
      numMissing = sum(cbData[i][:, j] .=== NaN)
      util.plot(util.density([cbData[i][:, j]; fill(-10, numMissing)], na=true), col="red",
                xlim=[-8,8], main="Y sample: $(i), marker: $(j), missing: $numMissing",
                bty="n", fg="grey")
      for iter in 1:length(y_imputed)
        yimp = y_imputed[iter]
        util.lines(util.density(yimp[i][:, j]), col=util.rgba("blue", .5))
      end
    end
  end
  R"par(mfrow=c(1,1))"
  util.devOff()


  idx_missing = [ findall(isnan.(cbData[i])) for i in 1:I ]
  idx = idx_missing[2][1]
  util.plotPdf("$(IMGDIR)/y_trace.pdf")
  util.hist([ y_imputed[b][2][idx] for b in 1:length(y_imputed) ], col="blue", border="transparent")
  util.plot([ y_imputed[b][2][idx] for b in 1:length(y_imputed) ], typ="l")
  util.devOff()

  for i in 1:I
    util.plotPng("$IMGDIR/y_imputed$(i).png")
    util.yZ_inspect(out[1], i=i, lastState.y_imputed, zlim=[-4,4], using_zero_index=false,
                    thresh=0.9, col=util.blueToRed(9))
    util.devOff()
  end

  for i in 1:I
    util.plotPng("$IMGDIR/y_dat$(i).png")
    util.yZ_inspect(out[1], i=i, cbData, zlim=[-4,4], using_zero_index=false, na="black",
                    thresh=0.9, col=util.blueToRed(9))
    util.devOff()
  end

  #= TODO: redo this with a thinned sample of gam
  # Plot QQ
  y_obs_range = util.y_obs_range(cbData)
  util.plotPdf("$IMGDIR/qq.pdf")
  println("Plotting QQ...")
  R"par(mfrow=c(3, 3), mar=c(5.1, 4, 2, 1))"
  for i in 1:I
    for j in 1:J
      println("i: $i, j: $j")
      # QQ of observed expression levels
      y_obs, y_pp = util.qq_yobs_postpred(cbData, i, j, out)
      util.myQQ(y_obs, y_pp, pch=20, ylab="post pred quantiles", xlab="y
                (observed) quantiles", main="i: $i, j: $j", xlim=y_obs_range,
                ylim=y_obs_range)
    end
  end
  R"par(mfrow=c(1, 1), mar=mar.default())"
  util.devOff()

  # Plot postpred hist
  util.plotPdf("$IMGDIR/postpred_hist.pdf")
  println("Plotting postpred histograms...")
  R"par(mfrow=c(3, 3), mar=c(5.1, 4, 2, 1))"
  for i in 1:I
    for j in 1:J
      println("i: $i, j: $j")
      # QQ of observed expression levels
      y_obs, y_pp = util.qq_yobs_postpred(cbData, i, j, out)
      util.hist(y_obs, prob=true, xlim=y_obs_range, xlab="", ylab="density",
                main="i: $i, j: $j", col=util.rgba("red", .3), border="transparent")
      util.hist(y_pp, prob=true, xlab="", ylab="", add=true, col=util.rgba("blue", .3),
                border="transparent")
    end
  end
  R"par(mfrow=c(1, 1), mar=mar.default())"
  util.devOff()
  =#
end
