import PyPlot
const plt = PyPlot
printKeys(x) = [print("$k  ") for k in keys(x)]
using Distributions
#=
plt.ioff() # turn off interactive graphics. turn on with plt.ion()
plt.ion() # turn on interactive graphics. turn on with plt.ion()
=#

function pyRange(n::Int)
  return collect(0:(n-1))
end

function saveimg(path::String)
  plt.savefig(path)
  plt.close()
end

function plotZ(Z::Matrix{T}; kw...) where {T <: Number}
  J, K = size(Z)
  cm = plt.cm_get_cmap(:Greys)
  img = plt.imshow(Z, aspect="auto", vmin=0, vmax=1, cmap=cm; kw...);
  ax = plt.gca()
  [ plt.axhline(y=i+.5, color="grey", linewidth=.5) for i in pyRange(J)];
  [ plt.axvline(x=i+.5, color="grey", linewidth=.5) for i in pyRange(K)];
  plt.yticks(pyRange(J), 1:J);
  plt.xticks(pyRange(K), 1:K, rotation=:vertical);
  return img
end

function plotY(Y::Matrix{T}; kw...) where {T  <: Number}
  cm = plt.cm_get_cmap(:bwr)
  cm[:set_under](color=:blue)
  cm[:set_over](color=:red)
  cm[:set_bad](color=:black)
  img = plt.imshow(Y, aspect="auto", cmap=cm; kw...) # vmin, vmax
  plt.plt[:colorbar]();
  return img
end

### IBP ###
J = 32
K = 8
Z = Int.(rand(J, K) .> .6);

plotZ(Matrix(Z'))
ax = plt.gca()
ax[:set_xlabel]("Features", fontsize=12)
ax[:set_ylabel]("Observations", fontsize=12)
plt.tight_layout()
saveimg("img/Z.pdf")

### Heatmap ###
Y = randn(30000, 32) * 2;
Y[:, 3] .= NaN;

plotY(Y, vmin=-3, vmax=3)
plt.tight_layout()
saveimg("img/heatmap.pdf")

# yZ plot
#plt.rc_context(Dict("axes.edgecolor" => "grey", "xtick.color" => "grey", "ytick.color" => "grey"))
plt.subplot2grid((1, 10), (0, 0), colspan=3)
plotZ(Matrix(Z))
plt.xticks(rotation=:horizontal)
plt.subplot2grid((1, 10), (0, 3), colspan=7)
plotY(Matrix(Y'), vmin=-3, vmax=3);
plt.yticks(pyRange(J), 1:J);
plt.xticks(rotation=:vertical)
plt.tight_layout()
saveimg("img/yZ.pdf")
#plt.rcdefaults()

# kde
function kde(x::Vector{T}; from::T=minimum(x), to::T=maximum(x), numPoints::Int=1000,
             bw::Float64=0.0, kernel::Function=z->pdf(Normal(), z)) where {T <: Number}
  n = length(x)
  if bw == 0.0
    bw = std(x) * n ^ (-0.2) * 1.06
  end
  x_grid = collect(range(from, stop=to, length=numPoints))
  return Dict(:dx => [sum(kernel.((xi .- x) / bw)) / (n * bw) for xi in x_grid], :x => x_grid)
end

# Density
dist = Gamma(3, 200)
@time x = rand(dist, 10000);
@time d = kde(x, numPoints=100, bw=100.);
plt.plot(d[:x], pdf.(dist, d[:x]), label=:truth, color=:blue)
plt.plot(d[:x], d[:dx], color=:orange, label=:kde)
x_ci = quantile(x, [.025, .975])
plt.fill_between(d[:x][x_ci[1] .< d[:x] .< x_ci[2]], d[:dx][x_ci[1] .< d[:x] .< x_ci[2]],
                color=:orange)
plt.tight_layout()
plt.legend()
saveimg("img/den.pdf")

# Histogram vs KDE
x = sort([rand(Normal(10, 1), 3000); rand(Normal(-10, 1), 3000)])
true_den = (pdf.(Normal(10, 1), x) .+ pdf.(Normal(-10, 1), x)) / 2
@time d = kde(x, numPoints=1000, from=-20., to=20., bw=.1)
@time d_auto = kde(x, numPoints=1000, from=-20., to=20.)
plt.plot(d[:x], d[:dx], color=:orange, label="kde: bw fixed at .1")
plt.plot(d_auto[:x], d_auto[:dx], color=:red, label="kde: auto-bw")
plt.plt[:hist](x, density=true, label=:hist, bins=50)
plt.plot(x, true_den, label="true density", c=:black)
plt.tight_layout()
plt.legend()
#plt.show()
saveimg("img/kde_vs_den.pdf")

# Axis everywhere
# https://matplotlib.org/examples/api/two_scales.html
fig, ax1 = plt.subplots()
ax1[:plot](randn(100))
ax1[:set_xlabel]("x")
# Make the y-axis label, ticks and tick labels match the line color.
ax1[:set_ylabel](:leftLabel, color=:blue)
ax1[:tick_params]("y", colors=:blue)

ax2 = ax1[:twinx]()
ax2[:set_ylabel](:blabla)
ax2[:tick_params](:blable)
saveimg("img/axisGalore.pdf")

# TODO: Plot in plot
# https://stackoverflow.com/questions/17458580/embedding-small-plots-inside-subplots-in-matplotlib
# https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplots.html
function infig_position(ax, rect)
  # Get current figure
  fig = plt.gcf()

  box = ax[:get_position]()
  inax_position = ax[:transAxes][:transform](rect[1:2])
  transFigure = fig[:transFigure][:inverted]()
  infig_pos = transFigure[:transform](inax_position)    
  width = box[:width] * rect[3]
  height = box[:height] * rect[4]
  xpos = infig_pos[1]
  ypos = infig_pos[2]

  return xpos, ypos, width, height
end

function plotPost(y, ax::PyPlot.PyObject; rect=[.7, .7, .3, .2], showTrace::Bool=true,
                  showAcc::Bool=true, digits_ci=2, numPoints=100, c=:royalblue,
                  alpha_level=.05, useHistogram=false, normalizeHist=true, histBins=0,
                  accFontsize=6, annfs=6)
  # Get current figure
  fig = plt.gcf()

  if showTrace
    # Get box position / dimensions
    xpos, ypos, width, height = infig_position(ax, rect)

    subax = fig[:add_axes]([xpos, ypos, width, height])
    subax[:axis](:off)
    subax[:plot](1:length(y), y, c=:grey, linewidth=.5)
    accRate = length(unique(y)) / length(y)
    if showAcc
      subax[:set_title]("acc: $(accRate*100)%", fontsize=accFontsize)
    end
  end

  # plot histogram / density
  if useHistogram
    # Not really supported
    if histBins == 0
      ax[:hist](y, density=normalizeHist)
    else
      ax[:hist](y, density=normalizeHist, bins=histBins)
    end
  else
    if length(unique(y)) == 1
      ax[:annotate]("mean: $(round(mean(y), digits=digits_ci))", color=:red,
                    xy=(.8, .6), fontsize=annfs, xycoords="axes fraction")
      ax[:annotate]("SD: $(round(std(y), digits=digits_ci))",
                    xy=(.8, .4), fontsize=annfs, xycoords="axes fraction")
    else
      # Confidence Interval
      a_lower = alpha_level / 2
      a_upper = 1 - a_lower
      y_ci = quantile(y, [a_lower, a_upper])
      d_ci = kde(y, numPoints=numPoints, from=y_ci[1], to=y_ci[2])
      d = kde(y, numPoints=numPoints)
      ax[:plot](d_ci[:x], d_ci[:dx], c=c)
      ax[:plot](d[:x], d[:dx], c=c, alpha=.5)
      ax[:fill_between](d[:x][y_ci[1] .< d[:x] .< y_ci[2]],
                        d[:dx][y_ci[1] .< d[:x] .< y_ci[2]], color=c)
      ax[:fill_between](d[:x][minimum(y) .< d[:x] .< maximum(y)],
                        d[:dx][minimum(y) .< d[:x] .< maximum(y)], color=c, alpha=.3)
      ax[:set_xticks](round.(y_ci, digits=digits_ci))
      y_mean = mean(y)
      ax[:vlines](y_mean, ymin=0, ymax=d[:dx][argmin(abs.(d[:x] .- y_mean))],
                  color=:red, linewidth=1)

      # Plot Annotations
      ax[:annotate]("mean: $(round(y_mean, digits=digits_ci))", color=:red,
                    xy=(.8, .6), fontsize=annfs, xycoords="axes fraction")
      ax[:annotate]("$(Int(round((1-alpha_level) * 100)))% CI", color=c,
                    xy=(.8, .5), fontsize=annfs, xycoords="axes fraction")
      ax[:annotate]("SD: $(round(std(y), digits=digits_ci))",
                    xy=(.8, .4), fontsize=annfs, xycoords="axes fraction")
    end
  end

  ax[:spines]["right"][:set_visible](false)
  ax[:spines]["top"][:set_visible](false)
  ax[:spines]["left"][:set_color]("grey")
  ax[:spines]["bottom"][:set_color]("grey")
end

function plotPost(y; kw...)
  fig, ax = plt.subplots()
  plotPost(y, ax; kw...)
  return fig, ax
end

#plotPost(Y[:,1], useHistogram=true, histBins=30)
#plotPost(Y[:,1])



function plotPosts(Y; use_tight_layout=true,
                   fig_size_inches=missing,
                   fig_subplots_adjust=missing,
                   kw...)
  J = size(Y, 2)
  fig, ax = plt.subplots(J,J)
  if use_tight_layout
    fig[:tight_layout]()
  end

  if !ismissing(fig_size_inches)
    fig[:set_size_inches](fig_size_inches[1], fig_size_inches[2])
  end

  if !ismissing(fig_subplots_adjust)
    fig[:subplots_adjust](wspace=fig_subplots_adjust[1],
                          hspace=fig_subplots_adjust[2])
  end

  plotPosts(Y, fig, ax; kw...)
end

function plotPosts(Y, fig, ax; details=false, digits_cor=2, scatter_alpha=.7, scatter_s=.5,
                   warn=true, kw...)
  J = size(Y, 2)
  counter = 0
  for c in 1:J
    for r in 1:J
      counter += 1
      if r == c
        plotPost(Y[:, r], ax[counter]; kw...)
      elseif r > c
        ax[counter][:axis](:off)
        #xpos, ypos, w, h = infig_position(ax[counter], [.5, .5, .5, .5])
        cor_ycr = round(cor(Y[:, c], Y[:, r]), digits=digits_cor)
        if !isnan(cor_ycr)
          ax[counter][:text](0, .4, "r = $cor_ycr", fontsize=abs(cor_ycr)*10+10)
        else
          if warn
            @warn "Correlation was NaN."
          end
          ax[counter][:text](0, .4, "r = ???", fontsize=10)
        end
      else
        #ax[counter][:plot](Y[:, c], Y[:, r], c=:grey, linewidth=.5)
        ax[counter][:scatter](Y[:, c], Y[:, r], c=:grey, alpha=scatter_alpha, s=scatter_s)
        if !details
          ax[counter][:axis](:off)
        end
      end
    end
  end
  return fig, ax
end

#Y = randn(300, 5)
#Y[:,1] .= Y[:, 5] .+ 10
s = [[-1., 0., -3., 2.] [0., .5, .8, .2] [-3., .8, .6, .2] [2, .2, .2, -1.]] 
S = s * s'
Y = rand(MvNormal([1,2,3,4], S), 1000)'
Y[:, 2] .= 2.0


fig, ax = plotPosts(Y, digits_ci=1, fig_subplots_adjust=(.2, .2), fig_size_inches=(8,8), annfs=6);
saveimg("img/plotPosts.pdf")

#= Or for more control
J = size(Y, 2)
fig, ax = plt.subplots(J,J);
fig[:tight_layout]()
fig[:set_size_inches](8, 8)
fig[:subplots_adjust](wspace=.1, hspace=.1)
plotPosts(Y, fig, ax, digits_ci=1);
saveimg("img/plotPosts.pdf")
=#

# Heatmap with annotations
W = rand(3, 10)
function plotW(W; digits_W=2, pad_colorbar=.05, shrink_colorbar=1, show_colorbar=true, kw...)
  I, K = size(W)
  img = plt.imshow(W, aspect=:auto; kw...)
  for i in pyRange(I)
    for k in pyRange(K)
      img[:axes][:text](k, i, round(W[i+1, k+1], digits=digits_W),
                        ha=:center, va=:center, color=:white)
    end
  end
  plt.xticks(pyRange(K), 1:K)
  plt.yticks(pyRange(I), 1:I)
  if show_colorbar
    plt.plt[:colorbar](pad=pad_colorbar, shrink=shrink_colorbar);
  end
  return img
end

img = plotW(W; pad_colorbar=.01, vmin=0., vmax=1.)
ax = plt.gca()
ax[:set_xlabel]("Features", fontsize=12)
ax[:set_ylabel]("Samples", fontsize=12)
plt.tight_layout()
saveimg("img/W.pdf")
