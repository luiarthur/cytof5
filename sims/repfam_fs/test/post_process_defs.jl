using Revise
using Cytof5
using Random
using Distributions
import PyPlot, PyCall
const plt = PyPlot.plt
using BSON
using RCall
@rimport rcommon

include("../Util/Util.jl")
include("simulatedata.jl")

#= Interactive plot
PyPlot.matplotlib.use("TkAgg")
=#

#= Non-interactive plot 
PyPlot.matplotlib.use("Agg")
=#

# Some python ploting setup
path_to_plot_defs = "../../vb"
pushfirst!(PyCall.PyVector(PyCall.pyimport("sys")."path"), "$path_to_plot_defs")
include("$(path_to_plot_defs)/post_process_defs_pyplot.jl")
VLIM = (-4, 4)

# Plot settings
# http://blog.juliusschulz.de/blog/ultimate-ipython-notebook
rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
rcParams["font.size"] = 15
rcParams["xtick.labelsize"] = 15
rcParams["ytick.labelsize"] = 15
plt.rcParams["figure.figsize"] = (6, 6)

#= Interactive plot
PyPlot.matplotlib.use("TkAgg")
=#

# Post process
extract(chain, sym) = [samp[sym] for samp in chain]
extract(sym) = [samp[sym] for samp in out[:samples][1]]
out = BSON.load("results/out_fs.bson")
simdat = BSON.load("results/data_fs.bson")[:simdat]
boxplot(x) = plt.boxplot(x, whis=[2.5, 97.5], showfliers=false, showmeans=true)

# Plot log likelihood
plt.plot(out[:loglike]); plt.xlabel("iter"); plt.ylabel("log-likelihood")

# Plot Wi
Ws = cat(extract(:theta__W)..., dims=3)
I = config[:dfs].data.I
plt.figure()
for i in 1:I
  plt.subplot(I, 1, i)
  boxplot(Ws[i, :, :]')
  plt.xlabel("cell phenotypes")
  plt.ylabel("W$(i)")
end
plt.tight_layout()

# Plot mus
deltas = extract(:theta__delta)
mus0 = Matrix(hcat([-cumsum(d[0]) for d in deltas]...)')
mus1 = Matrix(hcat([cumsum(d[1]) for d in deltas]...)')
mus = [mus0 mus1]
boxplot(mus)
plt.axhline(0)

# Plot sig2
sig2s = Matrix(hcat(extract(:theta__sig2)...)')
boxplot(sig2s)
plt.axhline(0)

rcommon.plotPosts(sig2s)

# lambda
lams = extract(:theta__lam)

# Plot Z
Zs_vec = extract(:theta__Z)
Zs = cat(Zs_vec..., dims=3)
plot_yz.plot_Z_only(Zs[:, :, end])

plot_yz.plot_Z_only(mean(Zs_vec))

i = 1
plot_yz.plot_Z(mean(Zs_vec), mean(Ws, dims=3)[i, :], lam[end][i],
               w_thresh=0.01, fs_markers=18, fs_celltypes=18, fs_lab=18)

i = 1
plot_yz.plot_Z(Zs_vec[end], mean(Ws, dims=3)[i, :], lam[end][i],
               w_thresh=0.01, fs_markers=18, fs_celltypes=18, fs_lab=18)
i = 2
plot_yz.plot_Z(Zs_vec[end], mean(Ws, dims=3)[i, :], lam[end][i],
               w_thresh=0.01, fs_markers=18, fs_celltypes=18, fs_lab=18)


# Plot eta
etas = extract(:theta__eta)
etas0 = [x[0] for x in etas]
etas1 = [x[1] for x in etas]
mean(etas0)
mean(etas1)
