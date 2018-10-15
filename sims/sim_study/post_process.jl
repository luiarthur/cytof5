using Distributions
using Cytof5, Random
using JLD2, FileIO
using Plots; pyplot()
using LaTeXStrings
#= Dependencies for Plots & pyplot
import Pkg
Pkg.add("Plots")
Pkg.add("PyCall")
Pkg.add("LaTeXStrings")
Pkg.add("StatPlots")

run(`pip install matplotlib`) # OR
run(`pip3 install matplotlib`) 
=#

const tex = latexstring

include("CytofImg.jl")
Plots.scalefontsizes(.8)


# TODO: REMOVE WHEN DONE
include("util.jl")
using RCall


OUTDIR = ARGS[1]
IMGDIR = "$OUTDIR/img/"
run(`mkdir -p $(IMGDIR)`)

println("Loading Data ...")
@load "$(OUTDIR)/output.jld2" out dat ll lastState c y_dat metrics

I, K = size(dat[:W])
K_MCMC = size(lastState.W, 2)
J = size(lastState.Z, 1)

# Plot loglikelihood
plot(ll[1000:end], ylab="log-likelihood", xlab="MCMC iteration", legend=:none, c=:grey20)
savefig("$(IMGDIR)/ll.pdf")

# Plot Z
Zpost = CytofImg.getPosterior(:Z, out[1])
Zmean = mean(Zpost)

CytofImg.plotZ(Zmean, colorbar=true, xlabel="Features", ylabel="Markers")
savefig("$IMGDIR/Z_mean.pdf")

Z_mean_est_leftordered = Cytof5.Model.leftOrder((Zmean .> .5)*1)
CytofImg.plotZ(Z_mean_est_leftordered, xlabel="Features", ylabel="Markers")
savefig("$IMGDIR/Z_mean_est_leftordered.pdf")

CytofImg.plotZ(dat[:Z], xlabel="Features", ylabel="Markers")
savefig("$IMGDIR/Z_true.pdf")

# Plot W
# Annotate heatmap
# https://discourse.julialang.org/t/annotations-and-line-widths-in-plots-jl-heatmaps/4259/2
Wpost = CytofImg.getPosterior(:W, out[1])
Wmean = mean(Wpost)

heatmap(Wmean, xlabel="Features", ylabel="Sampels", c=:viridis, legend=true,
        border=true, yticks=1:I, clim=(0, .3))
CytofImg.annotateHeatmap(Wmean)
savefig("$IMGDIR/W_mean.pdf")

heatmap(dat[:W], xlabel="Features", ylabel="Sampels", c=:viridis, legend=true,
        border=true, yticks=1:I, clim=(0, .3))
CytofImg.annotateHeatmap(Wmean)
savefig("$IMGDIR/W_true.pdf")

# Get lam
lamPost = CytofImg.getPosterior(:lam, out[1])
unique(lamPost)

# Get b0
b0Post = hcat(CytofImg.getPosterior(:b0, out[1])...)'
CytofImg.plotPosts(Matrix(b0Post), q_digits=2, useDensity=true, traceFont=font(5),
                   titles=[("truth=$b0") for b0 in dat[:b0]])
                   #titles=[tex("\$\\beta_{0$i}\$") for i in 1:I])
savefig("$IMGDIR/b0.pdf")

include("CytofImg.jl")
x = CytofImg.postSummary(Matrix(b0Post))

# Get b1
b1Post = hcat(CytofImg.getPosterior(:b1, out[1])...)'
CytofImg.plotPosts(Matrix(b1Post), q_digits=2, useDensity=true, traceFont=font(5),
                   titles=[tex("\$\\beta_{1$i}\$") for i in 1:I])
savefig("$IMGDIR/b1.pdf")

#= color between
x = range(0, stop=2*pi, length=100)
y1 = sin.(x)
y2 = sin.(x) .* exp.(-x/10)
Plots.plot(x, y1, legend=false, fill=y2)
plot(x, y1, legend=false, fill=(0, y2, :green))
=#



# Plot Posterior Prob of Missing
#include("CytofImg.jl");
layout=(I, 1)
plot(layout=layout, border=true, bordercolor=:lightgrey, grid=false)
for i in 1:I
  pmiss_mean, pmiss_lower, pmiss_upper, y_seq = CytofImg.postProbMiss(b0Post, b1Post, i)
  # FIXME: why do I need to add 5E-3?!
  Plots.plot!(y_seq, pmiss_lower, legend=false, title=latexstring("I=$i"), fill=pmiss_upper .+ 5E-3, subplot=i)
end
savefig("$IMGDIR/probMissPost.pdf")

# Get mus
mus0Post = hcat([m[:mus][0] for m in out[1]]...)'
mus1Post = hcat([m[:mus][1] for m in out[1]]...)'
musPost = [ mus0Post mus1Post ]

CytofImg.StatPlots.boxplot(musPost, ylabel=L"$\mu^*$", legend=false, xaxis=false, c=:steelblue,
                           grid=false, yrotation=90, border=true)
vline!([size(musPost, 2)/2 + .5], c=:grey30)
hline!([0], c=:grey30)
hline!(dat[:mus][0], line=:dot, c=:steelblue)
hline!(dat[:mus][1], line=:dot, c=:steelblue)
savefig("$IMGDIR/mus.pdf")

# Get sig2
sig2Post = hcat(CytofImg.getPosterior(:sig2, out[1])...)'
CytofImg.plotPosts(Matrix(sig2Post), titles= ["truth: $s2" for s2 in dat[:sig2]], showAccRate=false)
savefig("$IMGDIR/sig2.pdf")

# Posterior of y_imputed
util.plotPdf("$(IMGDIR)/ydatPost.pdf")
R"par(mfrow=c(4,2))"
for i in 1:I
  for j in 1:J
    util.plot(util.density(dat[:y][i][:, j], na=true), col="red", xlim=[-8,8],
              main="Y sample: $(i), marker: $(j)", bty="n", fg="grey")
    for iter in 1:length(y_imputed)
      yimp = y_imputed[iter]
      util.lines(util.density(yimp[i][:, j]), col=util.rgba("blue", .5))
    end
    util.lines(util.density(dat[:y_complete][i][:, j]), col="grey")
  end
end
R"par(mfrow=c(1,1))"
util.devOff()


idx_missing = [ findall(isnan.(y_dat.y[i])) for i in 1:y_dat.I ]
idx = idx_missing[2][1]
util.plotPdf("$(IMGDIR)/y_trace.pdf")
util.hist([ y_imputed[b][2][idx] for b in 1:length(y_imputed) ], col="blue", border="transparent")
util.plot([ y_imputed[b][2][idx] for b in 1:length(y_imputed) ], typ="l")
util.devOff()

# ARI - adjusted Rand Index ∈  (0, 1). Metric for clustering.
# Higher is better.
open("$IMGDIR/ari.txt", "w") do file
  ariCytof = [ x[1] for x in util.ari.(dat[:lam], lastState.lam) ]
  write(file, "ARI lam: $ariCytof\n")
end

#=
y_141 = [ yimp[1][4, 1] for yimp in y_imputed ]
R"hist"(y_141)
R"plot"(y_141, typ="l")
=#

for i in 1:I
  CytofImg.yZ_inspect(out[1], dat[:y], i, clim=(-6,6), ycolor=:bluesreds, marker_names=1:32)
  @time savefig("$IMGDIR/yDataSorted$(i).pdf")

  CytofImg.yZ_inspect(out[1], lastState.y_imputed, i, clim=(-6,6),
                      ycolor=:bluesreds, marker_names=1:32)
  @time savefig("$IMGDIR/y_imputed$(i).pdf")
end


#=
include("CytofImg.jl")
i=1
x = CytofImg.yZ_inspect(out[1], lastState.y_imputed, i, clim=(-6,6), ycolor=:bluesreds, marker_names=1:32)
sp2 = Plots.get_subplot(x, 2)
ax_y = Plots.get_axis(sp2, :y)
attr!(ax_y, :mirror);
=#

open("$IMGDIR/priorBeta.txt", "w") do file
  b0Prior = join(c.b0_prior, "\n")
  b1Prior = join(c.b1_prior, "\n")
  write(file, "b0Prior:\n$b0Prior\nb1Prior:\n$b1Prior\n")
end

