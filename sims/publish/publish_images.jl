using Distributions
using Cytof5, Random
using JLD2, FileIO
import BSON
include("salso.jl")

# Plotting
using PyCall
matplotlib = pyimport("matplotlib")
plt = pyimport("matplotlib.pyplot")
matplotlib.use("Agg")
# Load current dir
pushfirst!(PyVector(pyimport("sys")."path"), "../vb")
plot_yz = pyimport("plot_yz")
blue2red = pyimport("blue2red")

# Path to scratch
RESULTS_DIR = "/scratchdata/alui2/cytof/results/"

# y heatmap colorbar range
VLIM = (-4, 4)

# get posterior samples by symbol
getPosterior(sym::Symbol, monitor) = [m[sym] for m in monitor]

# read a single object from JLD2
function loadSingleObj(objPath)
  data = load(objPath)
  return data[collect(keys(data))[1]]
end

# plot y/z
function make_yz(y, Zs, Ws, lams, imgdir; w_thresh=.01, lw=3,
                 Z_true=nothing, fs_y=22, fs_z=22, fs_ycbar=22, fs_zcbar=22)
  mkpath(imgdir)
  I = length(y)
  for i in 1:I
    idx_best = estimate_ZWi_index(Zs, Ws, i)

    Zi = Int.(Zs[idx_best])
    Wi = Float64.(Ws[idx_best][i, :])
    lami = Int64.(lams[idx_best][i])
    yi = Float64.(y[i])

    # plot Yi, lami
    plt.figure(figsize=(8, 8))
    plot_yz.plot_y(yi, Wi, lami, vlim=VLIM, cm=blue2red.cm(9), lw=lw,
                   fs_xlab=fs_y, fs_ylab=fs_y, fs_lab=fs_y, fs_cbar=fs_ycbar)
    plt.savefig("$(imgdir)/y$(i).pdf", bbox_inches="tight")
    plt.close()

    # plot Zi, Wi
    plt.figure(figsize=(8, 8))
    plot_yz.plot_Z(Zi, Wi, lami, w_thresh=w_thresh, add_colorbar=false,
                   fs_lab=fs_z, fs_celltypes=fs_z, fs_markers=fs_z, fs_cbar=fs_zcbar)
    plt.savefig("$(imgdir)/Z$(i).pdf", bbox_inches="tight")
    plt.close()
  end

  if Z_true != nothing
    # plot Z true
    plt.figure(figsize=(8, 8))
    plot_yz.plot_Z_only(Z_true, fs=fs_z,
                        xlab="cell types", ylab="markers", rotate_xticks=false)
    plt.savefig("$(imgdir)/Z_true.pdf", bbox_inches="tight")
    plt.close()

    # plot ZT true
    plt.figure(figsize=(8, 8))
    plot_yz.plot_Z_only(Z_true', fs=fs_z,
                        xlab="markers", ylab="cell types")
    plt.savefig("$(imgdir)/ZT_true.pdf", bbox_inches="tight")
    plt.close()
  end
end

function make_yz(path_to_output)
  if occursin(".jld2", path_to_output)
    output = load(path_to_output)
    path_to_data = "$(dirname(path_to_output))/reduced_data/reduced_cb.jld2"
    y = loadSingleObj(path_to_data)
    out = output["out"]
  elseif occursin(".bson", path_to_output)
    output = BSON.load(path_to_output)
    y = output[:simdat][:y]
    out = output[:out]
  else
    println("Neither jld2 nor bson?!")
  end

  results_dir = "$(dirname(path_to_output))/img"
  results_dir = replace(results_dir, RESULTS_DIR => "results/")

  Zs = getPosterior(:Z, out[1])
  Ws = getPosterior(:W, out[1])
  lams = getPosterior(:lam, out[1])

  if haskey(output, :simdat)
    Z_true = output[:simdat][:Z]
  else
    Z_true = nothing
  end
  make_yz(y, Zs, Ws, lams, results_dir, Z_true=Z_true)
end

# MAIN
# path_to_output = "$(RESULTS_DIR)/cb/best/output.jld2"
@time for (root, dirs, files) in walkdir(RESULTS_DIR)
  for file in files
    if occursin("output.jld2", file) || occursin("output.bson", file)
    # if occursin("output.bson", file)
      path_to_output = "$(root)/$(file)"
      println("Current: $(path_to_output)")
      make_yz(path_to_output)
    end
  end
end


