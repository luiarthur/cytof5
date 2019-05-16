using Distributions
using Cytof5, Random, RCall
using JLD2, FileIO
import Printf
include("../sim_study/util.jl")
include("compress_data.jl")
Random.seed!(0)

function loadSingleObj(objPath)
  data = load(objPath)
  return data[collect(keys(data))[1]]
end

# TODO:
# Write a macro to reconstruct types 

# Turn reconstructed type into type State
function convertState(s)
  println("Trying to convert reconstructed state to Cytof5.Model.State ...")
  return Cytof5.Model.State(Z=s.Z, delta=s.delta, alpha=s.alpha, v=s.v, W=s.W,
                            sig2=s.sig2, eta=s.eta, lam=s.lam, gam=s.gam,
                            y_imputed=s.y_imputed)
end

# FIXME: similarity_Z...
function convertConstants(c)
  println("Trying to convert reconstructed state to Cytof5.Model.Constants ...")
  return Cytof5.Model.Constants(alpha_prior=c.alpha_prior,
                                delta_prior=c.delta_prior, W_prior=c.W_prior,
                                eta_prior=c.eta_prior, sig2_prior=c.sig2_prior,
                                beta=c.beta, K=c.K, L=c.L,
                                sig2_range=c.sig2_range,
                                # there should be new stuff here ...
                                probFlip_Z=c.probFlip_Z, similarity_Z=x -> x)

end

# TODO: Write somrthing generic here.
function convertType(s, T)
  if T == Cytof5.Model.State
    return convertState(s)
  elseif T == Cytof5.Model.Constants
    return convertConstants(s)
  else
    println("Don't know how to convert to type $(T)")
  end
end



function post_process(path_to_output, thresh=0.9, min_presences=[0, .01, .03, .05])
  jld_filename = split(path_to_output, "/")[end]
  outdir = join(split(path_to_output, "/")[1:end-1], "/")
  # datapath = "data/reduced_cb.jld2"
  datapath = "$outdir/reduced_data/reduced_cb.jld2"
  println("datapath: $datapath")
  IMGDIR = "$outdir/img/"

  run(`mkdir -p $(IMGDIR)/dden/`)

  println("Loading Data ...")
  cbData = loadSingleObj(datapath)
  cbData = decompress_data(cbData)
  println("Loading Results ...")

  # FIXME: Not efficient!
  init_is_defined = "init" in keys(load(path_to_output))
  println("init_is_defined = $init_is_defined ...")

  if init_is_defined
    @load path_to_output out ll lastState c init dden
  else
    @load path_to_output out ll lastState c
  end

  I, K_MCMC = size(lastState.W)
  J = size(lastState.Z, 1)
  N = size.(lastState.y_imputed, 1)

  # Set png resolution settings
  s_png = 10
  fy(clus) = util.addCut(clus, s_png)
  fZ(Z) = util.addGridLines(Z, s_png)

  println("Making eta")
  open("$IMGDIR/eta.txt", "w") do file
    # TODO
    eta = util.getPosterior(:eta, out[1])
    eta0_mean = mean([ei[0] for ei in eta])
    eta1_mean = mean([ei[1] for ei in eta])
    eta_mean = Dict(0 => eta0_mean, 1 => eta1_mean)

    header = "i", "j", "z", "l", "eta"
    write(file, "$(join(header, "     "))\n")
    for i in 1:I
      for j in 1:J
        for z in 0:1
          for l in 1:c.L[z]
            line = (i, j, z, l, eta_mean[z][i, j, l])
            line = Printf.@sprintf("%d %5d %5d %5d     %.5f", line...);
            write(file, "$line\n")
          end
        end
      end
    end
  end

  println("Making eta_obs.txt")
  open("$IMGDIR/eta_obs.txt", "w") do file
    header = "i", "j", "z", "l", "p"
    write(file, "$(join(header, "     "))\n")

    for i in 1:I
      for j in 1:J
        for z in 0:1
          for l in 1:c.L[z]
            idx_observed = util.idx_observed_ij(cbData, i, j)
            z_ij = out[1][end][:Z][j, out[1][end][:lam][i][idx_observed]]
            gam_ij = out[2][end][:gam][i][idx_observed, j]
            mij_sum = length(idx_observed)
            p = sum((gam_ij .== l) .& (z_ij .== z)) / mij_sum
            line = (i, j, z, l, p)
            line = Printf.@sprintf("%d %5d %5d %5d     %.5f", line...);
            write(file, "$line\n")
          end
        end
      end
    end
  end

  # Plot loglikelihood
  util.plotPdf("$(IMGDIR)/ll_complete_history.pdf")
  util.plot(ll, ylab="log-likelihood", xlab="MCMC iteration", typ="l");
  util.abline(v=length(ll) - length(out[1]))
  util.devOff()

  util.plotPdf("$(IMGDIR)/ll_postburn.pdf")
  util.plot(ll[end-length(out[1]):end], ylab="log-likelihood", xlab="MCMC iteration", typ="l");
  util.devOff()

  function addGridLines(J::Int, K::Int, col="grey", lwd=1)
    util.abline(v=(1:K) .+ .5, h=(1:J) .+ .5, col=col, lwd=lwd);
  end

  # Plot Z
  Zpost = util.getPosterior(:Z, out[1])
  Zmean = mean(Zpost)

  println("Making Z ...")
  try
    util.plotPdf("$IMGDIR/Z_mean.pdf")
    util.myImage(Zmean, xlab="Features", ylab="Markers", addL=true, col=util.greys(11),
                 f=Z->addGridLines(J, K_MCMC));
    util.devOff()

    util.plotPdf("$IMGDIR/Z_mean_est_leftordered.pdf")
    util.myImage(Cytof5.Model.leftOrder((Zmean .> .5)*1),
                 xlab="Features", ylab="Markers", addL=true, col=util.greys(11),
                 f=Z->addGridLines(J, K_MCMC));
    util.devOff()
  catch
    println("Failed to make complete making Z ...")
  end

  println("Plot Z diffs")
  if init_is_defined
    util.plotPdf("$IMGDIR/Z_minus_init.pdf")
    for i in 1:length(Zpost)
      z = Zpost[i]
      util.myImage(z - init.Z, xlab="Features", ylab="Markers", addL=true, col=util.blueToRed(3),
                   zlim=[-1, 1], f=Z->addGridLines(J, K_MCMC), main="Z_$i - Z_0");
    end
    util.devOff()
  else
    println("Can't make Z-diffs because init isn't defined ...")
    run(`rm -f $IMGDIR/Z_minus_init.pdf`)
  end

  # Plot Posterior Density for i, j observed
  for i in 1:I
    for j in 1:J
      println("Plot posterior density for (i: $i, j: $j) observed ...")
      util.plotPdf("$(IMGDIR)/dden/dden_i$(i)_j$(j).pdf")
      dyij = util.density(filter(yij -> !isnan(yij), cbData[i][:, j]))
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

  for i in 1:I
    println("Separate graphs for yZ $(i)...")
    idx_best = R"estimate_ZWi_index($(out[1]), $i)"[1]
    Zi = out[1][idx_best][:Z]
    Wi = out[1][idx_best][:W][i,:]

    # Point Est for Wi and Zi. txt and pdf.
    open("$IMGDIR/W_$(i)_hat.txt", "w") do file
      write(file, "$(join(Wi, "\n"))\n")
    end

    open("$IMGDIR/W$(i)_hat_ordered_cumsum.txt", "w") do file
      ord = sortperm(Wi, rev=true)
      cs_wi_sorted = cumsum(Wi[ord])
      write(file, "num_features,k,wi,cumprop\n")
      for k in 1:K_MCMC
        write(file, "$(k),$(ord[k]),$(Wi[ord][k]),$(cs_wi_sorted[k])\n")
      end
    end

    open("$IMGDIR/Z$(i)_hat.txt", "w") do file
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

      util.plotPng("$IMGDIR/y_dat$(i)_only_minpresence$(min_presence).png")
      ord_yi = sortperm(lami)
      util.myImage(cbData[i][ord_yi[1 .<= lami[ord_yi] .<= K_trunc], :],
                   addL=true, f=yi->util.addCut(lami, s_png),
                   zlim=[-4,4], col=util.blueToRed(9), na="black", xlab="markers",
                   ylab="cells");
      # util.myImage(cbData[i][sortperm(lami), :], addL=true, f=yi->util.addCut(lami),
      #              zlim=[-4,4], col=util.blueToRed(9), na="black", xlab="markers",
      #              ylab="cells");
      util.devOff()

      util.plotPdf("$IMGDIR/Z_hat$(i)_minpresence$(min_presence).pdf", w=5, h=10)
      util.myImage(Zi[:, common_celltypes], addL=false, ylab="markers", yaxt="n",
                   f=Z->addGridLines(J, K_trunc), xaxt="n", xlab="celltypes");

      perc = string.(round.(Wi[common_celltypes] * 100, digits=2), "%")
      R"""
      axis(3, at=1:$K_trunc, label=$(perc), las=1, fg="grey", cex.axis=1)
      axis(1, at=1:$K_trunc, label=$(common_celltypes), las=1,
           fg="grey", cex.axis=1)
      axis(2, at=1:$J, label=1:$J, las=2, fg="grey", cex.axis=1)
      """
      util.devOff()
    end
  end

  # Plot alpha
  alphaPost = util.getPosterior(:alpha, out[1])
  println("Making alpha ...")
  util.plotPdf("$(IMGDIR)/alpha.pdf")
  util.plotPost(alphaPost, ylab="density", xlab="alpha", main="");
  util.devOff()

  # Plot W
  Wpost = util.getPosterior(:W, out[1])
  W_mean = mean(Wpost)

  println("Making W...")
  util.plotPdf("$IMGDIR/W.pdf")

  cols = vcat([k .+ c.K .* (0:I-1) for k in 1:c.K]...)
  colnames = vec(["$i:$k" for i in 1:I, k in 1:c.K])
  W_post = hcat([hcat([w[i, :] for w in Wpost]...)' for i in 1:I]...)
  util.boxplot(W_post[:, cols], col=R"rep(2:$(I+1), $(c.K))", names=colnames, las=2, cex=.2);
  util.abline(v=(0:c.K) * I .+ .5, lty=2, col="grey")

  R"par(mfrow=c($I, 1), mar=c(5, 5.1, 0.5, 2.1))"
  for i in 1:I
    util.boxplot(hcat([w[i, :] for w in Wpost]...)', ylab="Posterior: W$i",
                 xlab=i<I ? "" : "Features",
                 col="steelblue", pch=20, cex=0);
  end
  R"par(mfrow=c(1, 1), mar=rcommon::mar.default())"
  util.devOff()

 
  # Missing Mechanism
  println("Making beta  ...")
  open("$IMGDIR/beta.txt", "w") do file
    for i in 1:I
      bi = join(c.beta[:, i], ", ")
      write(file, "for i=$i, beta = $bi \n")
    end
  end

  # y_imputed cache
  y_imputed = [ o[:y_imputed] for o in out[2] ]
  B = length(y_imputed)
  y_imputed_min = minimum(vcat([minimum.(yb) for yb in y_imputed]...))
  y_imputed_max = maximum(vcat([maximum.(yb) for yb in y_imputed]...))
  y_imputed_range = [y_imputed_min, y_imputed_max]

  # Plot missing mechanism
  println("Making prob miss  ...")
  util.plotPdf("$IMGDIR/prob_miss.pdf")
  R"par(mfrow=c($I, 1))"
  for i in 1:I
    util.plotProbMiss(c.beta, i, xlim=y_imputed_range)
  end
  R"par(mfrow=c(1,1))"
  util.devOff()

  # Get mus
  delta0Post = hcat([m[:delta][0] for m in out[1]]...)'
  delta1Post = hcat([m[:delta][1] for m in out[1]]...)'
  mus0Post = -cumsum(delta0Post, dims=2)
  mus1Post =  cumsum(delta1Post, dims=2)
  musPost = [ mus0Post mus1Post ]

  println("Making mus...")
  util.plotPdf("$IMGDIR/mus.pdf")
  util.boxplot(musPost, ylab="mu*", xlab="", xaxt="n", col="steelblue", pch=20, cex=0);
  util.abline(h=0, v=size(mus0Post, 2) + .5, col="grey30", lty=1);
  util.devOff()

  util.plotPdf("$IMGDIR/delta0.pdf")
  util.plotPosts(delta0Post);
  util.devOff()

  util.plotPdf("$IMGDIR/delta1.pdf")
  util.plotPosts(delta1Post);
  util.devOff()

  # Get sig2
  sig2Post = hcat(util.getPosterior(:sig2, out[1])...)'
  sig2Mean = mean(sig2Post, dims=1)
  sig2Sd = std(sig2Post, dims=1)

  println("Making sig2...")
  util.plotPdf("$IMGDIR/sig2.pdf")
  util.plotPosts(sig2Post);
  util.boxplot(sig2Post, ylab="sig2", xlab="sample", xaxt="n", col="steelblue", pch=20, cex=0);
  util.devOff()

  println("Making v...")
  vPost = hcat(util.getPosterior(:v, out[1])...)'
  util.plotPdf("$IMGDIR/v.pdf")
  util.boxplot(vPost, ylab="v", xlab="", xaxt="n", col="steelblue", pch=20, cex=0);
  util.devOff()

  util.plotPdf("$IMGDIR/v_each.pdf")
  for k in 1:c.K
    util.plotPost(vPost[:, k], main="v_$k")
  end
  util.devOff()

  println("Making v_cumprod...")
  util.plotPdf("$IMGDIR/v_cumprod.pdf")
  util.boxplot(cumprod(vPost, dims=2),
               ylab="v_cumprod", xlab="", xaxt="n", col="steelblue", pch=20, cex=0);
  util.devOff()


  # Posterior of eps
  println("Making eps...")
  epsPost = hcat(util.getPosterior(:eps, out[1])...)'
  util.plotPdf("$IMGDIR/eps.pdf")
  util.plotPosts(epsPost);
  util.devOff()

  # Posterior of y_imputed
  println("Making ydatPost.pdf ...")
  util.plotPdf("$(IMGDIR)/ydatPost.pdf")
  R"par(mfrow=c(4,2))"
  for i in 1:I
    for j in 1:J
      print("\r i: $i, j: $j")
      numMissing = sum(cbData[i][:, j] .=== NaN)
      h = maximum(util.density(y_imputed[1][i][:, j])[:y])
      util.plot(util.density([cbData[i][:, j]; fill(-10, numMissing)], na=true, bw=.3),
                col="red", xlim=[-8,8], main="Y sample: $(i), marker: $(j),
                missing: $numMissing", bty="n", fg="grey", ylim=[0,h])
      for iter in 1:B
        yimp = y_imputed[iter]
        util.lines(util.density(yimp[i][:, j]), col=util.rgba("blue", .5))
      end
    end
  end
  R"par(mfrow=c(1,1))"
  println()
  util.devOff()


  idx_missing = [ findall(isnan.(cbData[i])) for i in 1:I ]
  num_missing_per_sample= 3
  idx_missing = [shuffle!(idx_missing[i])[1:num_missing_per_sample] for i in 1:I]
  idx_missing = [map(idx -> (i, idx[1], idx[2]), idx_missing[i]) for i in 1:I]
  idx_missing = vcat(idx_missing...)
  for idx in idx_missing
    i, n, j = idx
    y_inj = [y_imputed[b][i][n, j] for b in 1:B]
    util.plotPdf("$(IMGDIR)/y_trace_i$(i)_n$(n)_j$(j).pdf")
    util.hist(y_inj, col="blue", border="transparent",
              main="", xlab="y (i: 2, n: $n, j: $j)", ylab="counts")
    util.plot(y_inj, typ="l", xlab="Index", ylab="y (i: $i, n: $n, j: $j)", main="")
    util.devOff()
  end

  for i in 1:I
    idx_missing = findall(isnan.(cbData[i]))
    y_i = [y_imputed[b][i][idx_missing] for b in 1:B]
    util.plotPdf("$IMGDIR/y_imputed_hist_i$(i).pdf")
    R"par(mfrow=c($I, 1), mar=c(5, 5.1, 0.5, 2.1))"
    util.hist(mean(y_i),    xlim=y_imputed_range, xlab="means of imputed y for sample $i", ylab="counts", main="");
    util.hist(maximum(y_i), xlim=y_imputed_range, xlab="max of imputed y for sample $i", ylab="counts", main="");
    util.hist(minimum(y_i), xlim=y_imputed_range, xlab="min of imputed y for sample $i", ylab="counts", main="");
    R"par(mfrow=c(1,1), mar=mar.default())"
    util.devOff()
  end


  for i in 1:I
    util.plotPng("$IMGDIR/y_imputed$(i).png")
    util.yZ_inspect(out[1], i=i, lastState.y_imputed, zlim=[-4,4], using_zero_index=false,
                    thresh=thresh, col=util.blueToRed(9), fy=fy, fZ=fZ)
    util.devOff()
  end

  for i in 1:I
    util.plotPng("$IMGDIR/y_dat$(i).png")
    util.yZ_inspect(out[1], i=i, cbData, zlim=[-4,4], using_zero_index=false, na="black",
                    thresh=thresh, col=util.blueToRed(9), fy=fy, fZ=fZ)
    util.devOff()
  end

  println("Making yZ with Z mean...")
  if any(lastState.eps .== 0)
    for i in 1:I
      util.plotPng("$IMGDIR/y_dat$(i)_with_zmean.png")

      idx_best = R"estimate_ZWi_index($(out[1]), $i)"[1]
      Zi = out[1][idx_best][:Z]
      Wi = out[1][idx_best][:W][i,:]
      lami = out[1][idx_best][:lam][i]

      S = [findall(lami .== k) for k in 1:c.K]
      Zi_bar = mean([[mean(o[:Z][j, o[:lam][i][S[k]]]) for j in 1:J, k in 1:c.K] for o in out[1]])

      util.yZ(cbData[i], Zi_bar, Wi, lami, zlim=[-4,4], thresh=thresh, col=util.blueToRed(9),
              na="black", using_zero_index=false, col_Z=R"grey(seq(1, 0, len=11))", 
              colorbar_Z=true, cex_z_leg=0.001, fy=fy, fZ=fZ)

      util.devOff()
    end
  else
    # Get lam
    println("Making lam...")
    lamPost = util.getPosterior(:lam, out[1])
    unique(lamPost)

    prop_lam0 = hcat([[mean(lam[i] .== 0) for lam in lamPost] for i in 1:I]...)
    sum_lam0 = hcat([[sum(lam[i] .== 0) for lam in lamPost] for i in 1:I]...)
    println("prop0 dim: $(size(prop_lam0))")
    util.plotPdf("$IMGDIR/lam0.pdf")
    util.boxplot(prop_lam0, ylab="P(lambda_i = 0 | data)",
                 xlab="sample", col="steelblue", pch=20, cex=0);
    util.devOff()

    util.plotPdf("$IMGDIR/lam0_counts.pdf")
    util.boxplot(sum_lam0, ylab="sum(lambda_i = 0 | data)",
                 xlab="sample", col="steelblue", pch=20, cex=0);
    util.devOff()


    # Plot lambda = 0 group
    println("Plot noisy group")
    for i in 1:I
      idx_best = R"estimate_ZWi_index($(out[1]), $i)"[1]
      lami = out[1][idx_best][:lam][i]
      idx0 = findall(lami .== 0)
      if length(idx0) > 0
        println("making y_dat$(i)_lam0.png")
        util.plotPng("$IMGDIR/y_dat$(i)_lam0.png")
        util.myImage(cbData[i][idx0, :], xlab="markers", ylab="cells", 
                     na="black", col=util.blueToRed(9), addL=true, zlim=[-4, 4], nticks=11)
        util.devOff()

        util.plotPng("$IMGDIR/y_imputed$(i)_lam0.png")
        util.myImage(lastState.y_imputed[i][idx0, :], xlab="markers", ylab="cells", 
                     na="black", col=util.blueToRed(9), addL=true, zlim=[-4, 4])
        util.devOff()
      else
        println("length(lam0) == 0. Not making y_dat$(i)_lam0.png!")
      end
    end
  end


  for i in 1:I
    println("Printing number of predominant celltypes for i = $i ...")
    idx_best = R"estimate_ZWi_index($(out[1]), $i)"[1]
    Wi = out[1][idx_best][:W][i,:]

    open("$IMGDIR/K_P$(round(Int, thresh * 100))_i$(i).txt", "w") do file
      cswi = cumsum(sort(Wi, rev=true))
      K_TOP = findfirst(cswi .> thresh)
      write(file, "K_TOP: $K_TOP \n")
    end
  end


  if init_is_defined
    println("Making y_dat init ...")
    for i in 1:I
      util.plotPng("$IMGDIR/y_dat$(i)_init.png")
      Zi = init.Z
      Wi = init.W[i, :]
      lami = init.lam[i]

      util.yZ(cbData[i], Zi, Wi, lami, zlim=[-4,4], thresh=thresh, col=util.blueToRed(9),
              na="black", using_zero_index=false, col_Z=R"grey(seq(1, 0, len=11))", 
              colorbar_Z=true, cex_z_leg=0.001, fy=fy, fZ=fZ)
      util.devOff()
    end
  else
    println("Skipping this plot, init is not defined ...")
  end
end
