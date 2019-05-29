using BSON
using PyCall
plt = pyimport("matplotlib.pyplot")

include("sample/sample_lam.jl")

pyrange(n) = collect(range(0, stop=n-1))

function add_gridlines_Z(Z)
  J, K = size(Z)
  for j in pyrange(J)
    plt.axhline(y=j+.5, color="grey", linewidth=.5)
  end

  for k in pyrange(K)
    plt.axvline(x=k+.5, color="grey", linewidth=.5)
  end
end

axhlines(x; kw...) = for xi in x plt.axhline(xi; kw...) end

function plot_Z(Z; colorbar=true)
  J, K = size(Z)
  p = plt.imshow(Z, aspect="auto", vmin=0, vmax=1, cmap=cm_greys)
  add_gridlines_Z(Z)
  plt.yticks(pyrange(J), pyrange(J) .+ 1, fontsize=10)
  plt.xticks(pyrange(K), pyrange(K) .+ 1, fontsize=10, rotation=90)
  if colorbar
    plt.colorbar()
  end
  return p
end

# For BSON
using Cytof5, Flux, Distributions

function post_process(output_path; thresh=.99)
  cm_greys = plt.cm.get_cmap("Greys", 5)
  VMIN, VMAX = VLIM = (-4, 4) 
  # cm = blue2red.cm(9)

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
  plt.plot(elbo[200:end])
  plt.xlabel("iteration")
  plt.ylabel("ELBO / sum(N)")
  plt.savefig("$(IMG_DIR)/elbo.pdf")

  # Metrics
  plt.figure()
  for (k, m) in metrics
    plt.plot(metrics[k][1:end]/sum(c.N))
    plt.xlabel("iteration")
    plt.ylabel(string(k))
  end
  plt.savefig("$(IMG_DIR)/metrics.pdf")

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
  for i in 1:c.I
    plt.hist(vec(y_samps[end][i][m[i]]))
    plt.xlim([-8, 1])
    plt.savefig("$(IMG_DIR)/y$(i)_imputed_hist.pdf")
  end

  # Prob miss
  ygrid = collect(range(-8, stop=0, length=500))
  pm = hcat([Cytof5.VB.prob_miss(ygrid, c.beta[i]...) for i in 1:c.I]...)
  plt.figure()
  for i in 1:c.I
    plt.plot(ygrid, pm[:, i], label="sample $i")
  end
  plt.legend()
  plt.xlabel("y")
  plt.ylabel("prob. of missing")
  plt.savefig("$(IMG_DIR)/prob_miss.pdf")

  # Z
  if c.use_stickbreak
    Z = [Int.(cumprod(reshape(s.v, 1, c.K)) .> s.H).data for s in samples]
  else
    Z = [Int.(reshape(s.v, 1, c.K) .> s.H).data for s in samples]
  end
  Z_mean = mean(Z)
  plot_Z(Z_mean)
  plt.savefig("$(IMG_DIR)/Z.pdf")

  # True Z
  if has_simdat
    plot_Z(simdat[:Z])
    plt.xlabel("features")
    plt.ylabel("markers")
    plt.savefig("$(IMG_DIR)/Z_true.pdf")

    plot_Z(simdat[:Z]')
    plt.ylabel("features")
    plt.xlabel("markers")
    plt.savefig("$(IMG_DIR)/ZT_true.pdf")
  end

  # W
  W = [s.W.data for s in samples]
  plt.figure()
  for i in 1:c.I
    plt.subplot(c.I, 1, i)
    Wi = hcat([w[i, :] for w in W]...)
    plt.boxplot(Wi', showmeans=true, whis=[2.5, 97.5], showfliers=false);
    plt.ylabel("W$i", rotation=0, labelpad=15)
    plt.xticks(rotation=90)
    if has_simdat
      for wik in simdat[:W][i, :]
        plt.axhline(wik, ls="--", lw=.5)
      end
    end
  end
  plt.tight_layout()
  # plt.show()
  plt.savefig("$(IMG_DIR)/W.pdf")

  # mu
  mu = hcat([[-cumsum(s.delta0.data); cumsum(s.delta1.data)] for s in samples]...)
  plt.boxplot(mu');
  if has_simdat
    plt.axhline(0)
    plt.axvline(c.L[0] + .5)
    for z in (0, 1)
      axhlines(simdat[:mus][z], color="grey", ls="--", lw=.5)
    end
  end
  plt.savefig("$(IMG_DIR)/mu.pdf")

  # sig2
  sig2 = hcat([s.sig2.data for s in samples]...)
  plt.boxplot(sig2');
  if has_simdat
    axhlines(simdat[:sig2], ls="--", color="grey", lw=.5);
  end
  plt.savefig("$(IMG_DIR)/sig2.pdf")

  # eps
  eps = hcat([s.eps.data for s in samples]...)
  plt.boxplot(eps');
  plt.savefig("$(IMG_DIR)/eps.pdf")

  # v
  v = hcat([s.v.data for s in samples]...)
  plt.boxplot(v');
  plt.savefig("$(IMG_DIR)/v.pdf")

  # v cumprod
  v_cumprod = hcat([cumprod(s.v.data) for s in samples]...)
  plt.boxplot(v_cumprod');
  plt.savefig("$(IMG_DIR)/v_cumprod.pdf")

  # alpha
  alpha = vcat([s.alpha.data for s in samples]...);
  plt.hist(alpha, density=true)
  plt.xlabel("alpha")
  plt.ylabel("density")
  plt.axvline(mean(alpha), color="black", linestyle="--")
  plt.axvline(quantile(alpha, .025), color="black", linestyle="--")
  plt.axvline(quantile(alpha, .975), color="black", linestyle="--")
  plt.savefig("$(IMG_DIR)/alpha.pdf")

  ### trace plots ###
  mkpath("$(IMG_DIR)/trace/")

  # TODO:
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
