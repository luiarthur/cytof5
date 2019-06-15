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
# multiple pages
PdfPages = pyimport("matplotlib.backends.backend_pdf").PdfPages

RESULTS_DIR = "/scratchdata/alui2/cytof/results/"
VLIM = (-4, 4)
getPosterior(sym::Symbol, monitor) = [m[sym] for m in monitor]
function loadSingleObj(objPath)
  data = load(objPath)
  return data[collect(keys(data))[1]]
end

# plot y/z
function make_yz(y, Zs, Ws, lams, dest; w_thresh=.01,
                 fs_y=18, fs_z=15, fs_ycbar=15, fs_zcbar=15)
  mkpath(dest)
  I = length(y)
  for i in 1:I
    idx_best = estimate_ZWi_index(Zs, Ws, i)
    Zi = Int.(Zs[idx_best])
    Wi = Ws[idx_best][i, :]
    lami = lams[idx_best][i]
    plot_yz.plot_y(y[i], Wi, lami, vlim=VLIM, cm=blue2red.cm(9),
                   fs_xlab=fs_y, fs_ylab=fs_y, fs_lab=fs_y, fs_cbar=fs_ycbar)
    plt.savefig("$(dest)/y$(i).pdf", bbox_inches="tight")
    plt.close()

    plot_yz.plot_Z(Zi, Wi, lami, w_thresh=w_thresh,
                   fs_lab=fs_z, fs_celltypes=fs_z, fs_markers=fs_z, fs_cbar=fs_zcbar)
    plt.savefig("$(dest)/Z$(i).pdf", bbox_inches="tight")
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
  make_yz(y, Zs, Ws, lams, results_dir)
end

# MAIN
# path_to_output = "$(RESULTS_DIR)/cb/best/output.jld2"
for (root, dirs, files) in walkdir(RESULTS_DIR)
  for file in files
    if occursin("output.jld2", file) || occursin("output.bson", file)
      path_to_output = "$(root)/$(file)"
      println("Current: $(path_to_output)")
      make_yz(path_to_output)
    end
  end
end


