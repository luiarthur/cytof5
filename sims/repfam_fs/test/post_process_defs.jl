using Revise
using Cytof5
using Random
using Distributions
using DelimitedFiles
import PyPlot, PyCall
const plt = PyPlot.plt
PyPlot.matplotlib.use("Agg")
using BSON
include("../../publish/salso.jl")

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

function post_process(path_to_output; path_to_simdat=nothing, vlim=(-4, 4),
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
  println("Number of MCMC samples: $(length(extract(:theta__delta)))")

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

  # Plot mus
  println("mus ...")
  deltas = extract(:theta__delta)
  mus0 = Matrix(hcat([-cumsum(d[0]) for d in deltas]...)')
  mus1 = Matrix(hcat([cumsum(d[1]) for d in deltas]...)')
  mus = [mus0 mus1]
  boxplot(mus)
  plt.axhline(0)
  plt.savefig("$(img_path)/mus.pdf", bbox_inches="tight")
  plt.close()

  # Plot sig2
  println("sig2 ...")
  sig2s = Matrix(hcat(extract(:theta__sig2)...)')
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
      ygrid = out[:c].y_grid
    else
      ygrid = collect(range(-10, stop=4, length=100))
    end

    for i in 1:I
      for j in 1:J
        dden_ij = [ddij[i, j] for ddij in dden_vec]
        dden_ij_mean = mean(dden_ij)
        dden_ij_lower = [quantile([ddij[g] for ddij in dden_ij], .025)
                         for g in 1:length(ygrid)]
        dden_ij_upper = [quantile([ddij[g] for ddij in dden_ij], .975)
                         for g in 1:length(ygrid)]
        plt.plot(ygrid, dden_ij_mean, color="blue")
        plt.fill_between(ygrid, dden_ij_lower, dden_ij_upper, alpha=.5,
                         color="blue")
        plt.xlabel("expression level")
        plt.ylabel("density")
        if dden_xlim != nothing
          plt.xlim(dden_xlim)
        end
        plt.savefig("$(img_path)/dden/dden_i$(i)_j$(j).pdf",
                    bbox_inches="tight")
        plt.close()
      end
    end
  end

  println("Done!")
end

#= Quick test
path_to_results = "results/test-sims-2/KMCMC5/z2/scale10"
path_to_output = "$(path_to_results)/output.bson"
path_to_simdat = "$(path_to_results)/simdat.bson"
post_process(path_to_output, path_to_simdat=path_to_simdat)
=#
