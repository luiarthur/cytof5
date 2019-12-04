using Revise
using Cytof5
using Random
using Distributions
using DelimitedFiles
import DataFrames
const DF = DataFrames
import CSV
import PyPlot, PyCall, Seaborn
const plt = PyPlot.plt
const sns = Seaborn
PyPlot.matplotlib.use("Agg")
using BSON
include("../../publish/salso.jl")
include("dden_complete.jl")

#= Interactive plot
PyPlot.matplotlib.use("TkAgg")
=#

#= Non-interactive plot 
PyPlot.matplotlib.use("Agg")
=#

# Load python defs
path_to_plot_defs = "../../vb"
pushfirst!(PyCall.PyVector(PyCall.pyimport("sys")."path"), "$path_to_plot_defs")
plot_yz = PyCall.pyimport("plot_yz")
blue2red = PyCall.pyimport("blue2red")
pyrange(n) = collect(range(0, stop=n-1))

# General plot settings
rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
rcParams["font.size"] = 15
rcParams["xtick.labelsize"] = 15
rcParams["ytick.labelsize"] = 15
rcParams["figure.figsize"] = (6, 6)

skipnan(x) = x[.!isnan.(x)]

function quantiles(X, q; dims, drop=false)
  Q = mapslices(x -> quantile(x, q), X, dims=dims)
  out = drop ? dropdims(Q, dims=dims) : Q
  return out
end

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
  plt.yticks(pyrange(J), pyrange(J) .+ 1, fontsize=rcParams["font.size"])
  plt.xticks(pyrange(K), pyrange(K) .+ 1, fontsize=rcParams["font.size"],
             rotation=90)
  if colorbar
    plt.colorbar()
  end
  return p
end

# plot y/z
function make_yz(y, Zs, Ws, lams, imgdir; vlim, 
                 w_thresh=.01, lw=3,
                 Z_true=nothing, 
                 fs_y=rcParams["font.size"],
                 fs_z=rcParams["font.size"],
                 fs_ycbar=rcParams["font.size"],
                 fs_zcbar=rcParams["font.size"])
  # Make img dir if needed
  mkpath(imgdir)
  mkpath("$(imgdir)/txt")

  I = length(y)
  for i in 1:I
    idx_best = estimate_ZWi_index(Zs, Ws, i)

    Zi = Int.(Zs[idx_best])
    Wi = Float64.(Ws[idx_best][i, :])
    lami = Int64.(lams[idx_best][i])

    # Write best idx
    open("$(imgdir)/txt/best_idx_$(i).txt", "w") do io
      println(io, idx_best)
    end

    # Write Zi
    open("$(imgdir)/txt/Z$(i)_hat.txt", "w") do io
      writedlm(io, Zi)
    end

    # Write Wi
    open("$(imgdir)/txt/W$(i)_hat.txt", "w") do io
      writedlm(io, Wi)
    end

    # Write lami
    open("$(imgdir)/txt/lam$(i)_hat.txt", "w") do io
      writedlm(io, lami)
    end

    yi = Float64.(y[i])

    # plot Yi, lami
    plt.figure(figsize=(6, 6))
    plot_yz.plot_y(yi, Wi, lami, vlim=vlim, cm=blue2red.cm(9), lw=lw,
                   fs_xlab=fs_y, fs_ylab=fs_y, fs_lab=fs_y, fs_cbar=fs_ycbar)
    plt.savefig("$(imgdir)/y$(i).pdf", bbox_inches="tight")
    plt.close()

    # plot Zi, Wi
    plt.figure(figsize=(6, 6))
    plot_yz.plot_Z(Zi, Wi, lami, w_thresh=w_thresh, add_colorbar=false,
                   fs_lab=fs_z, fs_celltypes=fs_z, fs_markers=fs_z,
                   fs_cbar=fs_zcbar)
    plt.savefig("$(imgdir)/Z$(i).pdf", bbox_inches="tight")
    plt.close()
  end

  if Z_true != nothing
    # plot Z true
    plt.figure(figsize=(6, 6))
    plot_yz.plot_Z_only(Z_true, fs=fs_z,
                        xlab="cell phenotypes", ylab="markers",
                        rotate_xticks=false)
    plt.savefig("$(imgdir)/Z_true.pdf", bbox_inches="tight")
    plt.close()

    # plot ZT true
    plt.figure(figsize=(6, 6))
    plot_yz.plot_Z_only(Z_true', fs=fs_z,
                        xlab="markers", ylab="cell phenotype")
    plt.savefig("$(imgdir)/ZT_true.pdf", bbox_inches="tight")
    plt.close()
  end
end


getpath(x) = join(split(x, "/")[1:end-1], "/")

function post_process(path_to_output;
                      path_to_simdat=nothing,
                      path_to_dat=nothing,
                      vlim=(-4, 4),
                      w_thresh=.01, dden_xlim=[-6, 6])
  results_path = getpath(path_to_output)

  # Define path to put images
  img_path = "$(results_path)/img"
  # Create dir if needed
  mkpath(img_path)
  mkpath("$(img_path)/txt")

  # Load sim output
  out = BSON.load(path_to_output)

  # Load sim data
  if path_to_simdat != nothing
    simdat = BSON.load(path_to_simdat)[:simdat]
  else
    simdat = nothing
  end

  # Define extraction functions
  extract(chain, sym) = [samp[sym] for samp in chain]
  extract(sym) = extract(out[:samples][1], sym)

  # Get number of samples: I
  I = length(out[:lastState].theta.y_imputed)
  # Get number of markers: J
  J = size(out[:lastState].theta.y_imputed[1], 2)

  # Print number of samples
  nsamps = length(extract(:theta__delta))
  println("Number of MCMC samples: $(nsamps)")

  # Plot log likelihood
  println("loglike ...")
  plt.plot(out[:loglike])
  plt.xlabel("iter")
  plt.ylabel("log-likelihood")
  plt.savefig("$(img_path)/loglike.pdf", bbox_inches="tight")
  plt.close()

  nburn = out[:nburn]
  plt.plot(out[:loglike][(nburn + 1):end])
  plt.xlabel("iter (post-burn)")
  plt.ylabel("log-likelihood")
  plt.savefig("$(img_path)/loglike_postburn.pdf", bbox_inches="tight")
  plt.close()


  # Plot W_star
  println("W_star ...")
  Wstar_vec = extract(:W_star)
  Wstar = cat(Wstar_vec..., dims=3)
  plt.figure()
  for i in 1:I
    plt.subplot(I, 1, i)
    boxplot(Wstar[i, :, :]')
    plt.xlabel("cell phenotypes")
    plt.ylabel("W*$(i)")
  end
  plt.tight_layout()
  plt.savefig("$(img_path)/Wstar.pdf", bbox_inches="tight")
  plt.close()

  open("$(img_path)/txt/Wstar_mean.txt", "w") do io
    writedlm(io, mean(Wstar_vec))
  end


  # Plot Wi
  println("W ...")
  Ws_vec = extract(:theta__W)
  Ws = cat(Ws_vec..., dims=3)
  plt.figure()
  for i in 1:I
    plt.subplot(I, 1, i)
    boxplot(Ws[i, :, :]')
    plt.xlabel("cell phenotypes")
    plt.ylabel("W$(i)")
    if simdat != nothing
      axhlines(simdat[:W][i, :])
    end
  end
  plt.tight_layout()
  plt.savefig("$(img_path)/W.pdf", bbox_inches="tight")
  plt.close()

  open("$(img_path)/txt/W_mean.txt", "w") do io
    writedlm(io, mean(Ws_vec))
  end

  # Plot alpha
  println("alpha ...")
  alphas = extract(:theta__alpha)
  plt.hist(alphas, density=true)
  plt.xlabel("alpha")
  plt.ylabel("density")
  plt.savefig("$(img_path)/alpha.pdf", bbox_inches="tight")
  plt.close()

  open("$(img_path)/txt/alpha_mean.txt", "w") do io
    writedlm(io, mean(alphas))
  end

  # Plot v
  println("v ...")
  v_vec = extract(:theta__v)
  v = hcat(v_vec...)

  open("$(img_path)/txt/v_mean.txt", "w") do io
    writedlm(io, mean(v_vec))
  end

  boxplot(v')
  plt.xlabel("cell phenotypes")
  plt.ylabel("v")
  plt.savefig("$(img_path)/v.pdf", bbox_inches="tight")
  plt.close()

  # Print R
  println("r ...")
  rs_vec = extract(:r)
  rs = cat(rs_vec..., dims=3)  # I x K x NMCMC
  Rs = dropdims(sum(rs, dims=2), dims=2)  # I x NMCMC

  Rs_df = DF.DataFrame(mean=vec(mean(Rs, dims=2)),
                       p_02_5=vec(quantiles(Rs, .025, dims=2)),
                       p_25_0=vec(quantiles(Rs, .250, dims=2)),
                       p_50_0=vec(quantiles(Rs, .500, dims=2)),
                       p_75_0=vec(quantiles(Rs, .750, dims=2)),
                       p_97_5=vec(quantiles(Rs, .975, dims=2)))
  CSV.write("$(img_path)/txt/Rs.csv", Rs_df)

  # Probability Ri equals each K
  prob_R_equals_K = [mean(Rs .== K, dims=2) for K in 1:maximum(Rs)]
  println(prob_R_equals_K)
  open("$(img_path)/txt/prob_R_equals_K.txt", "w") do io
    writedlm(io, prob_R_equals_K)
  end

  # Plot mus
  println("mus ...")
  deltas = extract(:theta__delta)
  mus_vec = [Dict(0 => -cumsum(delta[0]),
                  1 => cumsum(delta[1]))
             for delta in deltas]
  mus0 = Matrix(hcat([-cumsum(d[0]) for d in deltas]...)')
  mus1 = Matrix(hcat([cumsum(d[1]) for d in deltas]...)')

  open("$(img_path)/txt/mus0_mean.txt", "w") do io
    writedlm(io, mean(mus0, dims=1))
  end

  open("$(img_path)/txt/mus1_mean.txt", "w") do io
    writedlm(io, mean(mus1, dims=1))
  end

  mus = [mus0 mus1]
  boxplot(mus)
  plt.axhline(0)
  plt.savefig("$(img_path)/mus.pdf", bbox_inches="tight")
  plt.close()

  # Plot delta
  println("delta ...")
  delta0 = Matrix(hcat([d[0] for d in deltas]...)')  # B x L0
  delta1 = Matrix(hcat([d[1] for d in deltas]...)')  # B x L1

  open("$(img_path)/txt/delta0_mean.txt", "w") do io
    writedlm(io, mean(delta0, dims=1))
  end

  open("$(img_path)/txt/delta1_mean.txt", "w") do io
    writedlm(io, mean(delta1, dims=1))
  end

  boxplot(delta0)
  plt.savefig("$(img_path)/delta0.pdf", bbox_inches="tight")
  plt.close()

  boxplot(delta1)
  plt.savefig("$(img_path)/delta1.pdf", bbox_inches="tight")
  plt.close()


  # Plot sig2
  println("sig2 ...")
  sig2s = Matrix(hcat(extract(:theta__sig2)...)')

  open("$(img_path)/txt/sig2_mean.txt", "w") do io
    writedlm(io, mean(sig2s, dims=1))
  end

  boxplot(sig2s)
  if simdat != nothing
    axhlines(simdat[:sig2])
  end
  plt.savefig("$(img_path)/sig2.pdf", bbox_inches="tight")
  plt.close()

  # Plot Z
  println("Z ...")
  Zs_vec = extract(:theta__Z)
  Zs = cat(Zs_vec..., dims=3)
  plot_Z(mean(Zs_vec))
  plt.savefig("$(img_path)/Zmean.pdf", bbox_inches="tight")
  plt.close()

  open("$(img_path)/txt/Z_mean.txt", "w") do io
    writedlm(io, mean(Zs_vec))
  end

  # Plot eta
  etas = extract(:theta__eta)
  etas0 = [x[0] for x in etas]
  etas1 = [x[1] for x in etas]
  mean(etas0)
  mean(etas1)

  # lambda
  lams = extract(:theta__lam)

  # eta
  println("eta ...")
  eta_vec = extract(:theta__eta)
  mkpath("$(img_path)/txt/eta")


  # y/z plots
  if simdat != nothing
    println("Making y/z plots ...")
    make_yz(simdat[:y], Zs_vec, Ws_vec, lams, "$(img_path)/yz",
            vlim=vlim, w_thresh=w_thresh, lw=3, Z_true=simdat[:Z])
  else
    println("Not implemented! No y/z plots ...")
  end

  # dden
  if :dden in keys(out)
    println("Making dden ...")
    # Create directory for the data densities
    mkpath("$(img_path)/dden")

    # Get dden
    dden_vec = out[:dden]

    # Ygrid
    if :c in keys(out)
      ygrid = out[:c].constants.y_grid
    else
      ygrid = collect(range(-10, stop=4, length=100))
    end

    for i in 1:I
      for j in 1:J
        print("\r i: $i j: $j  ")
        dden_ij = [ddij[i, j] for ddij in dden_vec]
        dden_ij_lower = [quantile([ddij[g] for ddij in dden_ij], .025)
                         for g in 1:length(ygrid)]
        dden_ij_upper = [quantile([ddij[g] for ddij in dden_ij], .975)
                         for g in 1:length(ygrid)]

        p_ci_obs = plt.fill_between(ygrid, dden_ij_lower, dden_ij_upper,
                                     alpha=.5, color="blue")


        plt.xlabel("expression level")
        plt.ylabel("density")
        if dden_xlim != nothing
          plt.xlim(dden_xlim)
        end

        # Add eta to posterior dden plot
        eta0_ij_mean = mean(eta[0][i, j, :] for eta in eta_vec)
        L0 = length(eta0_ij_mean)
        plt.scatter(mean(mus0, dims=1), zeros(L0), 
                    s=eta0_ij_mean * 60 .+ 10, marker="X", color="green")

        open("$(img_path)/txt/eta/eta0_i$(i)_j$(j)_mean.txt", "w") do io
          writedlm(io, eta0_ij_mean)
        end

        eta1_ij_mean = mean(eta[1][i, j, :] for eta in eta_vec)
        L1 = length(eta1_ij_mean)
        p_mu = plt.scatter(mean(mus1, dims=1), zeros(L1),
                           s=eta1_ij_mean * 60 .+ 10, marker="X",
                           color="green")

        open("$(img_path)/txt/eta/eta1_i$(i)_j$(j)_mean.txt", "w") do io
          writedlm(io, eta1_ij_mean)
        end
 
        if simdat != nothing
          # Plot simulated data truth
          p_yobs = sns.kdeplot(skipnan(simdat[:y][i][:, j]),
                               color="red", bw=.1, label="tmp")
          p_yobs = p_yobs.get_legend_handles_labels()[1][1]

          # Histogram of observed data only
          # plt.hist(skipnan(simdat[:y][i][:, j]), color="red",
          #          alpha=.3, label="y (observed)",
          #          density=true, bins=30)

          if :eta in keys(simdat)
            eta_true = simdat[:eta]
          else
            eta_true = Dict(0 => ones(I, J, simdat[:L][0]),
                            1 => ones(I, J, simdat[:L][1]))
          end

          dgrid = dden_complete(ygrid, simdat[:W], eta_true,
                                simdat[:Z], simdat[:mus],
                                simdat[:sig2], i=i, j=j)

          p_truth_complete, = plt.plot(ygrid, dgrid, color="grey", ls="--")
        else
          # TODO: Plot histogram of data
        end

        # Plot complete posterior density (obs and imputed)
        dd_complete_post = [dden_complete(ygrid, Ws_vec[b],
                                          eta_vec[b],
                                          Zs_vec[b],
                                          mus_vec[b],
                                          sig2s[b, :], i=i, j=j)
                            for b in 1:nsamps]

        dd_complete_post = hcat(dd_complete_post...)  # legnth(ygird) x nsamps
        dcp_lower = quantiles(dd_complete_post, .025, dims=2, drop=true)
        dcp_upper = quantiles(dd_complete_post, .975, dims=2, drop=true)
        p_ci_complete = plt.fill_between(ygrid, dcp_lower, dcp_upper, alpha=.5,
                                         color="orange")

        # TODO: PICK UP HERE
        plt.legend([p_truth_complete, p_ci_complete,
                    p_yobs, p_ci_obs, p_mu],
                   ["truth (complete)", "95% CI (complete)",
                    "y (obs)", "95% CI (obs)", PyPlot.L"$\mu^\star$"])
        plt.savefig("$(img_path)/dden/dden_i$(i)_j$(j).pdf",
                    bbox_inches="tight")
        plt.close()
      end
    end
    println()

  end

  println("Done!")
end

#= Quick test
path_to_results = "results/test-sims-3/KMCMC5/z2/scale10"
path_to_output = "$(path_to_results)/output.bson"
path_to_simdat = "$(path_to_results)/simdat.bson"
post_process(path_to_output, path_to_simdat=path_to_simdat)
=#
