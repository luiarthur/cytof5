using Revise
using Cytof5
using Random
using Distributions
import PyPlot, PyCall
const plt = PyPlot.plt
PyPlot.matplotlib.use("Agg")
using BSON

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
plt.rcParams["figure.figsize"] = (6, 6)

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

getpath(x) = join(split(x, "/")[1:end-1], "/")

function post_process(path_to_output; vlim=(-4, 4))
  results_path = getpath(path_to_output)
  path_to_simdat = "$(results_path)/simdat.bson"

  # Define path to put images
  img_path = "$(results_path)/img"
  # Create dir if needed
  mkpath(img_path)

  # Load sim output
  out = BSON.load(path_to_output)

  # Load sim data
  simdat = BSON.load(path_to_simdat)[:simdat]

  # Define extraction functions
  extract(chain, sym) = [samp[sym] for samp in chain]
  extract(sym) = extract(out[:samples][1], sym)

  # Plot log likelihood
  println("loglike ...")
  plt.plot(out[:loglike]); plt.xlabel("iter"); plt.ylabel("log-likelihood")
  plt.savefig("$(img_path)/loglike.pdf")
  plt.close()

  # Plot Wi
  println("W ...")
  Ws = cat(extract(:theta__W)..., dims=3)
  I = length(simdat[:y])
  plt.figure()
  for i in 1:I
    plt.subplot(I, 1, i)
    boxplot(Ws[i, :, :]')
    plt.xlabel("cell phenotypes")
    plt.ylabel("W$(i)")
    axhlines(simdat[:W][i, :])
  end
  plt.tight_layout()
  plt.savefig("$(img_path)/W.pdf")
  plt.close()

  # Plot mus
  println("mus ...")
  deltas = extract(:theta__delta)
  mus0 = Matrix(hcat([-cumsum(d[0]) for d in deltas]...)')
  mus1 = Matrix(hcat([cumsum(d[1]) for d in deltas]...)')
  mus = [mus0 mus1]
  boxplot(mus)
  plt.axhline(0)
  plt.savefig("$(img_path)/mus.pdf")
  plt.close()

  # Plot sig2
  println("sig2 ...")
  sig2s = Matrix(hcat(extract(:theta__sig2)...)')
  boxplot(sig2s)
  plt.axhline(0)
  plt.savefig("$(img_path)/sig2.pdf")
  plt.close()

  # Plot Z
  println("Z ...")
  Zs_vec = extract(:theta__Z)
  Zs = cat(Zs_vec..., dims=3)
  plot_Z(mean(Zs_vec))
  plt.savefig("$(img_path)/Zmean.pdf")
  plt.close()

  # Plot eta
  etas = extract(:theta__eta)
  etas0 = [x[0] for x in etas]
  etas1 = [x[1] for x in etas]
  mean(etas0)
  mean(etas1)

  # lambda
  lams = extract(:theta__lam)

  println("Done!")
end

#=
post_process("results/test-sims/KMCMC2/z1/scale0/output.bson")
=#
