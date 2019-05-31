using BSON
using PyCall
plt = pyimport("matplotlib.pyplot")

# Load current dir
pushfirst!(PyVector(pyimport("sys")."path"), "")
plot_yz = pyimport("plot_yz").plot_yz
blue2red = pyimport("blue2red")

# multiple pages
PdfPages = pyimport("matplotlib.backends.backend_pdf").PdfPages

include("sample/sample_lam.jl")

pyrange(n) = collect(range(0, stop=n-1))
function boxplot(x; showmeans=true, whis=[2.5, 97.5], showfliers=false, kw...)
  plt.boxplot(x, showmeans=showmeans, whis=whis, showfliers=showfliers; kw...)
end

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
cm_greys = plt.cm.get_cmap("Greys", 5)

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

function post_process(output_path; thresh=.99, w_thresh=.01)
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

  println("metrics...")
  # ELBO
  plt.plot(elbo[200:end])
  plt.xlabel("iteration")
  plt.ylabel("ELBO / sum(N)")
  plt.savefig("$(IMG_DIR)/elbo.pdf")
  plt.close()

  # Metrics
  plt.figure()
  begin 
    local counter = 0
    for (k, m) in metrics
      counter += 1
      plt.subplot(length(metrics), 1, counter)
      plt.plot(metrics[k][1:end]/sum(c.N))
      if counter == length(metrics)
        plt.xlabel("iteration")
      end
      plt.ylabel(string(k))
    end
    plt.tight_layout()
    plt.savefig("$(IMG_DIR)/metrics.pdf")
  end
  plt.close()

  NSAMPS = 200
  samples = [Cytof5.VB.rsample(state)[2] for n in 1:NSAMPS]
  trace = [Cytof5.VB.rsample(s)[2] for s in state_hist]

  println("y samples ...")
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
    plt.close()
  end

  println("prob miss...")
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
  plt.close()

  # Z
  println("Z ...")
  if c.use_stickbreak
    Z = [Int.(cumprod(reshape(s.v, 1, c.K)) .> s.H).data for s in samples]
  else
    Z = [Int.(reshape(s.v, 1, c.K) .> s.H).data for s in samples]
  end
  Z_mean = mean(Z)
  plot_Z(Z_mean)
  plt.savefig("$(IMG_DIR)/Z.pdf")
  plt.close()

  # True Z
  if has_simdat
    plot_Z(simdat[:Z])
    plt.xlabel("features")
    plt.ylabel("markers")
    plt.savefig("$(IMG_DIR)/Z_true.pdf")
    plt.close()

    plot_Z(simdat[:Z]')
    plt.ylabel("features")
    plt.xlabel("markers")
    plt.savefig("$(IMG_DIR)/ZT_true.pdf")
    plt.close()
  end

  # W
  println("W ...")
  W = [s.W.data for s in samples]
  plt.figure()
  for i in 1:c.I
    plt.subplot(c.I, 1, i)
    Wi = hcat([w[i, :] for w in W]...)
    boxplot(Wi')
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
  plt.close()

  # mu
  println("mu ...")
  mu = hcat([[-cumsum(s.delta0.data); cumsum(s.delta1.data)] for s in samples]...)
  boxplot(mu');
  plt.axhline(0, alpha=.5)
  plt.axvline(c.L[0] + .5, alpha=.5)
  if has_simdat
    for z in (0, 1)
      axhlines(simdat[:mus][z], color="grey", ls="--", lw=.5)
    end
  end
  plt.savefig("$(IMG_DIR)/mu.pdf")
  plt.close()
  
  # sig2
  println("sig2 ...")
  sig2 = hcat([s.sig2.data for s in samples]...)
  boxplot(sig2');
  if has_simdat
    axhlines(simdat[:sig2], ls="--", color="grey", lw=.5);
  end
  plt.savefig("$(IMG_DIR)/sig2.pdf")
  plt.close()

  # eps
  println("eps ...")
  eps = hcat([s.eps.data for s in samples]...)
  boxplot(eps');
  plt.savefig("$(IMG_DIR)/eps.pdf")
  plt.close()

  # v
  println("v ...")
  v = hcat([s.v.data for s in samples]...)
  boxplot(v');
  plt.savefig("$(IMG_DIR)/v.pdf")
  plt.close()

  # v cumprod
  v_cumprod = hcat([cumprod(s.v.data) for s in samples]...)
  boxplot(v_cumprod');
  plt.savefig("$(IMG_DIR)/v_cumprod.pdf")
  plt.close()

  # alpha
  println("alpha ...")
  alpha = vcat([s.alpha.data for s in samples]...);
  plt.hist(alpha, density=true)
  plt.xlabel("alpha")
  plt.ylabel("density")
  plt.axvline(mean(alpha), color="black", linestyle="--")
  plt.axvline(quantile(alpha, .025), color="black", linestyle="--")
  plt.axvline(quantile(alpha, .975), color="black", linestyle="--")
  plt.savefig("$(IMG_DIR)/alpha.pdf")
  plt.close()

  ### trace plots ###
  println("traces ... ")
  mkpath("$(IMG_DIR)/trace/")

  # TODO:
  # Z trace
  # if c.use_stickbreak
  #   Z_trace = [Int.(reshape(cumprod(t.v), 1, c.K) .> t.H).data for t in trace]
  # else
  #   Z_trace = [Int.(reshape(t.v, 1, c.K) .> t.H).data for t in trace]
  # end
  # pdf_pages = PdfPages("$(IMG_DIR)/trace/Z.pdf")
  # for z in Z_trace
  #   fig = plt.figure()
  #   fig = plot_Z(z, fig)
  #   fig.xlabel("features")
  #   fig.ylabel("markers")
  #   pdf_pages.savefig(fig)
  # end
  # pdf_pages.close()

  # mu_trace
  mu_trace = hcat([[-cumsum(s.delta0.data); cumsum(s.delta1.data)] for s in trace]...)
  plt.plot(mu_trace')
  plt.xlabel("iteration")
  plt.ylabel("mu")
  plt.savefig("$(IMG_DIR)/trace/mu.pdf")
  plt.close()

  # sig2_trace
  sig2_trace = hcat([s.sig2.data for s in trace]...)
  plt.plot(sig2_trace')
  plt.xlabel("iter")
  plt.ylabel("sig2")
  plt.savefig("$(IMG_DIR)/trace/sig2.pdf")
  plt.close()

  # eps_trace
  eps_trace = hcat([s.eps.data for s in trace]...)
  plt.plot(eps_trace')
  plt.xlabel("iter")
  plt.ylabel("eps")
  plt.savefig("$(IMG_DIR)/trace/eps.pdf")
  plt.close()

  # W trace
  W_trace = cat([s.W.data for s in trace]..., dims=3)
  for i in 1:c.I
    plt.plot(W_trace[i, :, :]')
    plt.xlabel("iter")
    plt.ylabel("W$i")
    plt.savefig("$(IMG_DIR)/trace/W$(i).pdf")
    plt.close()
  end

  # v trace
  v_trace = hcat([s.v.data for s in trace]...)
  plt.plot(v_trace')
  plt.xlabel("iter")
  plt.ylabel("v")
  plt.savefig("$(IMG_DIR)/trace/v.pdf")
  plt.close()

  # alpha trace
  alpha_trace = vcat([s.alpha.data for s in trace]...)
  plt.plot(alpha_trace)
  plt.xlabel("iter")
  plt.ylabel("alpha")
  plt.savefig("$(IMG_DIR)/trace/alpha.pdf")
  plt.close()

  ### yZ ###
  println("yz ... ")
  @time if has_simdat
    lam = [sample_lam(state, simdat[:y], c) for b in 1:30]
  else
    lam = [sample_lam(state, y, c) for b in 1:30]
  end
  lam_mode = lam_f(lam, Distributions.mode)
  # println(lam_mode[1][1:10])
  W_mean= dropdims(mean(cat(W..., dims=3), dims=3), dims=3)

  mkpath("$(IMG_DIR)/yz")
  for i in 1:c.I
    # Yi
    # lami_est, k_ord = relabel_lam(lam_mode[i], W_mean[i, :])
    lami_est, k_ord = relabel_lam2(lam_mode[i], W_mean[i, :])
    if has_simdat
      yi = simdat[:y][i]
    else
      yi = y[i]
    end
    plt.imshow(yi[sortperm(lami_est), :], aspect="auto",
               vmin=VMIN, vmax=VMAX, cmap=blue2red.cm(9))
    plt.colorbar()
    plt.xlabel("markers")
    plt.ylabel("cells")
    plt.savefig("$(IMG_DIR)/yz/y$(i)_post.pdf")
    plt.close()

    # Zi
    # k_top = argmax(cumsum(W_mean[i, k_ord]) .> thresh)
    # plot_Z(Z_mean[:, k_ord[1:k_top]]')
    k_common = W_mean[i, k_ord] .> w_thresh
    plot_Z(Z_mean[:, k_ord[k_common]]')
    plt.xlabel("markers")
    plt.ylabel("cell types")
    plt.savefig("$(IMG_DIR)/yz/Z$(i)_post.pdf")
    plt.close()

    # yz_i
    plt.figure(figsize=(8,8))
    # plot_yz(yi, Z_mean, W_mean[i, :], lam_mode[i], w_thresh=w_thresh,
    plot_yz(yi, Z_mean, W_mean[i, :], lam_mode[i], w_thresh=w_thresh,
            cm_y=blue2red.cm(9), vlim_y=VLIM)
    plt.savefig("$(IMG_DIR)/yz/yz$(i)_post.pdf", dpi=500)
    plt.close()
  end
end
