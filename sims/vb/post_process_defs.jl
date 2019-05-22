using BSON
using RCall
@rimport graphics as plt
@rimport grDevices as dev
@rimport cytof3
@rimport rcommon
@rimport base
include("sample/sample_lam.jl")

function plotpng(fname, s=10, w=480, h=480, ps=12; kw...)
  dev.png(fname, w=w*s, h=h*s, pointsize=ps*s, kw...)
end
addcut(clus, s=1) = plt.abline(h=base.cumsum(base.table(clus)) .+ .5, lwd=3*s, col="yellow")

# For BSON
using Cytof5, Flux, Distributions

function post_process(output_path; thresh=.99)
  out = BSON.load(output_path)
  if :simdat in keys(out)
    has_simdat = true
    simdat = decompress_simdat!(out[:simdat])
  else
    has_simdat = false
    y = Matrix{Float64}.(out[:y])
  end
  RESULTS_DIR = join(split(output_path, "/")[1:end-1], "/")
  IMG_DIR = "$(RESULTS_DIR)/img/"
  mkpath(IMG_DIR)

  c = out[:c]
  elbo = out[:metrics][:elbo] / sum(c.N)
  state = out[:state]
  state_hist = out[:state_hist]
  metrics = out[:metrics]

  # ELBO
  dev.pdf("$(IMG_DIR)/elbo.pdf")
  plt.plot(elbo[200:end], xlab="iteration", ylab="ELBO / sum(N)", typ="l")
  dev.dev_off()

  # Metrics
  dev.pdf("$(IMG_DIR)/metrics.pdf")
  plt.par(mfrow=[length(metrics), 1], oma=rcommon.oma_ts(), mar=rcommon.mar_ts())
  for (k, m) in metrics
    plt.plot(metrics[k][1:end]/sum(c.N), xlab="iter", ylab=string(k), typ="l",
             xaxt="n", las=2)
  end
  plt.axis(1)
  plt.par(mfrow=[1, 1], oma=rcommon.oma_default(), mar=rcommon.mar_ts())
  dev.dev_off()

  NSAMPS = 200
  samples = [Cytof5.VB.rsample(state)[2] for n in 1:NSAMPS]
  trace = [Cytof5.VB.rsample(s)[2] for s in state_hist]

  # y_samps
  if has_simdat
    m = [isnan.(yi) for yi in simdat[:y]]
  else
    m = [isnan.(yi) for yi in y]
  end
  if has_simdat
    y_samps = [Tracker.data.(Cytof5.VB.rsample(state, simdat[:y], c)[3]) for n in 1:10]
  else
    y_samps = [Tracker.data.(Cytof5.VB.rsample(state, y, c)[3]) for n in 1:10]
  end
  dev.pdf("$(IMG_DIR)/y_hist.pdf")
  for i in 1:c.I
    plt.hist(vec(y_samps[end][i][m[i]]), xlab="", ylab="", main="");
  end
  dev.dev_off()

  # Prob miss
  ygrid = collect(range(-8, stop=0, length=500))
  pm = hcat([Cytof5.VB.prob_miss(ygrid, c.beta[i]...) for i in 1:c.I]...)
  col=2:4
  dev.pdf("$(IMG_DIR)/prob_miss.pdf")
  plt.matplot(ygrid, pm, xlab="y", ylab="prob. of missing", lw=3, typ="l", lty=1, col=col)
  plt.legend("topright", legend=1:3, col=col, lwd=3)
  dev.dev_off()

  # Z
  if c.use_stickbreak
    Z = [Int.(cumprod(reshape(s.v, 1, c.K)) .> s.H) for s in samples]
  else
    Z = [Int.(reshape(s.v, 1, c.K) .> s.H) for s in samples]
  end
  Z_mean = mean(Z).data
  dev.pdf("$(IMG_DIR)/Z.pdf")
  fZ(z, s_png=1) = plt.abline(h=collect(1:size(z, 1)) .+ .5,
                              v=collect(1:size(z, 2)) .+ .5, col="grey", lwd=s_png)
  cytof3.my_image(Z_mean, xlab="features", ylab="markers",
                  col=cytof3.greys(10), addL=true, f=fZ)
  dev.dev_off()

  # True Z
  if has_simdat
    dev.pdf("$(IMG_DIR)/Z_true.pdf")
    cytof3.my_image(simdat[:Z], xlab="features", ylab="markers", col=cytof3.greys(10), addL=true, f=fZ)
    dev.dev_off()
  end

  # TODO: Add truth

  # W
  W = [s.W.data for s in samples]
  dev.pdf("$(IMG_DIR)/W.pdf")
  for i in 1:c.I
    Wi = hcat([w[i, :] for w in W]...)
    plt.boxplot(Wi');
    if has_simdat
      plt.abline(h=simdat[:W][i, :], lty=2, col="grey")
    end
  end
  dev.dev_off()

  # mu
  mu = hcat([[-cumsum(s.delta0.data); cumsum(s.delta1.data)] for s in samples]...)
  dev.pdf("$(IMG_DIR)/mu.pdf")
  plt.boxplot(mu');
  plt.abline(h=0, v=c.L[0]+.5, col="grey");
  if has_simdat
    plt.abline(h=simdat[:mus][0], lty=2, col="grey");
    plt.abline(h=simdat[:mus][1], lty=2, col="grey");
  end
  dev.dev_off()

  # sig2
  sig2 = hcat([s.sig2.data for s in samples]...)
  dev.pdf("$(IMG_DIR)/sig2.pdf")
  plt.boxplot(sig2');
  if has_simdat
    plt.abline(h=simdat[:sig2], lty=2, col="grey");
  end
  dev.dev_off()

  # eps
  eps = hcat([s.eps.data for s in samples]...)
  dev.pdf("$(IMG_DIR)/eps.pdf")
  plt.boxplot(eps');
  dev.dev_off()

  # v
  v = hcat([s.v.data for s in samples]...)
  dev.pdf("$(IMG_DIR)/v.pdf")
  plt.boxplot(v');
  dev.dev_off()

  # v cumprod
  v_cumprod = hcat([cumprod(s.v.data) for s in samples]...)
  dev.pdf("$(IMG_DIR)/v_cumprod.pdf")
  plt.boxplot(v_cumprod');
  dev.dev_off()

  # alpha
  alpha = vcat([s.alpha.data for s in samples]...);
  dev.pdf("$(IMG_DIR)/alpha.pdf")
  rcommon.plotPost(alpha, main="alpha");
  dev.dev_off()

  ### trace plots ###
  mkpath("$(IMG_DIR)/trace/")

  # Z trace
  if c.use_stickbreak
    Z_trace = [Int.(reshape(cumprod(t.v), 1, c.K) .> t.H).data for t in trace]
  else
    Z_trace = [Int.(reshape(t.v, 1, c.K) .> t.H).data for t in trace]
  end
  dev.pdf("$(IMG_DIR)/trace/Z.pdf")
  for z in Z_trace
    cytof3.my_image(z, xlab="features", ylab="markers")
    plt.abline(h=collect(1:c.J) .+ .5, v=collect(1:c.K) .+ .5, col="grey")
  end
  dev.dev_off()

  # mu_trace
  mu_trace = hcat([[-cumsum(s.delta0.data); cumsum(s.delta1.data)] for s in trace]...)
  dev.pdf("$(IMG_DIR)/trace/mu.pdf")
  plt.matplot(mu_trace', xlab="iter", ylab="mu", typ="l", lwd=2)
  dev.dev_off()

  # sig2_trace
  sig2_trace = hcat([s.sig2.data for s in trace]...)
  dev.pdf("$(IMG_DIR)/trace/sig2.pdf")
  plt.matplot(sig2_trace', xlab="iter", ylab="sig2", typ="l", lwd=2)
  dev.dev_off()

  # eps_trace
  eps_trace = hcat([s.eps.data for s in trace]...)
  dev.pdf("$(IMG_DIR)/trace/eps.pdf")
  plt.matplot(eps_trace', xlab="iter", ylab="eps", typ="l", lwd=2)
  dev.dev_off()

  # W trace
  W_trace = cat([s.W.data for s in trace]..., dims=3)
  for i in 1:c.I
    dev.pdf("$(IMG_DIR)/trace/W$(i).pdf")
    plt.matplot(W_trace[i, :, :]', xlab="iter", ylab="W$(i)", typ="l", lwd=2)
    dev.dev_off()
  end

  # v trace
  v_trace = hcat([s.v.data for s in trace]...)
  dev.pdf("$(IMG_DIR)/trace/v.pdf")
  plt.matplot(v_trace', xlab="iter", ylab="v", typ="l", lwd=2)
  dev.dev_off()

  # alpha trace
  alpha_trace = vcat([s.alpha.data for s in trace]...)
  dev.pdf("$(IMG_DIR)/trace/alpha.pdf")
  plt.plot(alpha_trace, xlab="iter", ylab="alpha", typ="l", lwd=2)
  dev.dev_off()

  ### yZ ###
  if has_simdat
    lam = [sample_lam(state, simdat[:y], c) for b in 1:10]
  else
    lam = [sample_lam(state, y, c) for b in 1:10]
  end
  lam_mode = lam_f(lam, mode)
  W_mean= dropdims(mean(cat(W..., dims=3), dims=3), dims=3)

  mkpath("$(IMG_DIR)/yz")
  s_png = 10
  for i in 1:c.I
    # Yi
    plotpng("$(IMG_DIR)/yz/y$(i)_post.png", s_png)
    lami_est, k_ord = relabel_lam(lam_mode[i], W_mean[i, :])
    if has_simdat
      yi = simdat[:y][i]
    else
      yi = y[i]
    end
    cytof3.my_image(yi[sortperm(lami_est), :], na="black",
                    zlim=[-4,4], col=cytof3.blueToRed(9),
                    f=x->addcut(lami_est, s_png),
                    addL=true, xlab="markers", ylab="cells");
    dev.dev_off()

    # Zi
    dev.pdf("$(IMG_DIR)/yz/Z$(i)_post.pdf")
    k_top = argmax(cumsum(W_mean[i, k_ord]) .> thresh)
    cytof3.my_image(Z_mean[:, k_ord[1:k_top]]', f=z->fZ(z, 1), addL=true,
                    col=cytof3.greys(10), xlab="markers", ylab="cell types")
    dev.dev_off()
  end
end
