module Plot

using Distributions
import PyPlot
const plt = PyPlot

# For Debugging
printKeys(x) = [print("$k  ") for k in keys(x)]

#=
plt.ioff() # turn off interactive graphics. turn on with plt.ion()
plt.ion() # turn on interactive graphics. turn on with plt.ion()
=#

function pyRange(n::Int)
  return collect(0:(n-1))
end

function plotZ(Z::Matrix; xticks=true, kw...)
  J, K = size(Z)
  cm = plt.cm_get_cmap(:Greys, 5)
  img = plt.imshow(Z, aspect="auto", vmin=0, vmax=1, cmap=cm; kw...);
  [ plt.axhline(y=i+.5, color="grey", linewidth=.5) for i in pyRange(J)];
  [ plt.axvline(x=i+.5, color="grey", linewidth=.5) for i in pyRange(K)];
  plt.yticks(pyRange(J), 1:J);
  if xticks
    plt.xticks(pyRange(K), 1:K, rotation=:vertical);
  end
  ax = plt.gca()
  ax[:tick_params](length=0)
  return ax
end

# Define BlueGreyRed ColorMap
let
  cm_name = :BlueGreyRed
  c = Matrix([[0, 0, 1] [.9, .9, .9] [1, 0, 0]]')
  cm = Plot.plt.ColorMap(c)
  Plot.plt.register_cmap(cm_name, cm)
end

function plotY(Y::Matrix; colorbar=true,
               cm=PyPlot.get_cmap(:BlueGreyRed, 9), vmin=-4, vmax=4, kw...)
  cm[:set_under](color=:blue)
  cm[:set_over](color=:red)
  cm[:set_bad](color=:black)
  img = plt.imshow(Y, aspect="auto", cmap=cm, vmin=vmin, vmax=vmax; kw...) # vmin, vmax
  if colorbar
    plt.plt[:colorbar]();
  end
  ax = plt.gca()
  ax[:tick_params](length=0)
  return img
end


"""
J, K = 20, 4
Y = randn(1000, J)
Y[:, 2] .= NaN
Z = Int.(randn(J, K) .> 0)
# Plot.plotY(Y)

Plot.plotYZ(Y, Z)
Plot.plt.savefig("yZ.pdf", bbox_inches="tight")
Plot.plt.close()
"""
function plotYZ(Y::Matrix, Z::Matrix, vmin=-4, vmax=4, y_ytick_fs=5, z_ytick_fs=5; kw...)
  J, K = size(Z)

  # Y
  plt.subplot2grid((10, 1), (0, 0), rowspan=7)
  plotY(Matrix(Y), colorbar=false, vmin=vmin, vmax=vmax);
  plt.yticks(fontsize=y_ytick_fs)
  plt.plt[:colorbar]()
  plt.xticks(pyRange(J), 1:J, rotation=:vertical);

  # Z
  plt.subplot2grid((10, 1), (8, 0), rowspan=3)
  ax = plotZ(Matrix(Z'))
  plt.xticks([])
  plt.yticks(fontsize=z_ytick_fs)
  # Add ticks to other axis
  ax = plt.gca()
  ax2 = ax[:twinx]()
  plt.plt[:colorbar](aspect=5)
  # Change default ticks! Comment this out to see the difference.
  ax2[:set_yticks](pyRange(K))
  plt.yticks((K-1) / K * pyRange(K) .+ .5, (1:K) / 10, fontsize=5)
  ax2[:tick_params](length=0)
end

# kde
"""
# Example:
dist = Gamma(3, 200)
x = rand(dist, 10000);
d = kde(x, numPoints=100, bw=100.);

plt.plot(d[:x], pdf.(dist, d[:x]), label=:truth, color=:blue)
plt.plot(d[:x], d[:dx], color=:orange, label=:kde)
x_ci = quantile(x, [.025, .975])
plt.fill_between(d[:x][x_ci[1] .< d[:x] .< x_ci[2]], d[:dx][x_ci[1] .< d[:x] .< x_ci[2]],
                color=:orange)
plt.tight_layout()
plt.legend()
"""
function kde(x::Vector; from=minimum(x), to=maximum(x), numPoints::Int=1000,
             bw::Float64=0.0, kernel::Function=z->pdf(Normal(), z))
  n = length(x)
  if bw == 0.0
    bw = std(x) * n ^ (-0.2) * 1.06
  end
  x_grid = collect(range(from, stop=to, length=numPoints))
  return Dict(:dx => [sum(kernel.((xi .- x) / bw)) / (n * bw) for xi in x_grid], :x => x_grid)
end

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
                  accFontsize=6, annfs=6, hide_yaxis=true)
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

  if hide_yaxis
    ax[:axes][:yaxis][:set_ticklabels]([])
    ax[:spines]["left"][:set_visible](false)
    ax[:tick_params](length=0, axis="y")
  end

  ax[:spines]["bottom"][:set_color]("grey")
end

function plotPost(y; kw...)
  fig, ax = plt.subplots()
  plotPost(y, ax; kw...)
  return fig, ax
end


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

end # Plot
