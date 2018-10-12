module CytofImg

using Plots
pyplot()
using StatPlots
using Distributions # std, mean, quantile

startupMsg = """
To scale font size globally for plots (for presentations mostly), do this:
```
julia> Plots.scalefontsizes(2) # scales all texts by factor of 2

# To reset
julia> Plots.reset_defaults()
```
"""
println(startupMsg)

function getPosterior(sym::Symbol, monitor)
  return [ m[sym] for m in monitor]
end


"""
turn on legend with `legend=true`

Note that xlabel / ylabel can be supplied by either:

```
julia> xlabel!("my xlab")
# OR
julia> plotZ(Z, xlabel="my xlab")
```

"""
function plotZ(Z::Matrix{T}; kw...) where {T <: Number}
  img = heatmap(Z, c=:Greys, legend=:none, border=true, bordercolor=:lightgrey; kw...)
  J, K = size(Z)
  if J > 1
    hline!((2:J) .- .5, c=:lightgrey)
  end
  if K > 1
    vline!((2:K) .- .5, c=:lightgrey)
  end
  return img
end

function plotY_heatmap(y::Matrix{Float64}; clim=(-5, 5), c=:pu_or)
  img = heatmap(y, c=c, clim=clim, background_color_inside=:black);
  return img
end

function plotYZ(y::Matrix{Float64}, Z::Matrix{Float64}; clim=(-5, 5))
  heatY = plotY_heatmap(y)
  heatZ = plotZ(Z)
  img = plot(hH, hZ, layout=(2, 1))
  return (img, heatY, heatZ)
end

function annotateHeatmap(mat::Matrix{T}; digits=2, textsize=6) where T
  rows, cols = size(mat)
  for c in 1:cols
    for r in 1:rows
      annotate!([(float(c), float(r),
                  text("$(round(mat[r,c], digits=digits))", textsize))])
    end
  end
end

function plotPost(x::Vector{T}; a::Float64=0.05, q_digits::Int=3, sd_digits::Int=3, add=false,
                  trace::Bool=true, useDensity::Bool=true,
                  parent=1, offset=1, traceFont=font(16), kw...) where {T <: Number}
  denColor = (0, .99, :steelblue)
  if useDensity
    if add
      img = StatPlots.density!(x, label="", grid=false, yaxis=false,
                               linecolor=:transparent,
                               fill=denColor, bordercolor=:transparent; kw...)
    else
      img = StatPlots.density(x, label="", grid=false, bordercolor=:transparent, yaxis=false,
                              fill=denColor, linecolor=:transparent; kw...)
    end
  else
    if add
      img = histogram!(x, label="", normed=true, grid=false,
                       bordercolor=:white, yaxis=false,
                       linecolor=:transparent, c=:steelblue; kw...)
    else
      img = histogram(x, label="", normed=true, grid=false,
                      bordercolor=:white, yaxis=false,
                      linecolor=:transparent, c=:steelblue; kw...)
    end
  end


  if trace
    accRate = length(unique(x)) / length(x)
    #                                x    y   W   H
    plot!(x, inset_subplots=(parent, bbox(0, .1, .3, .3, :top, :right)), subplot=parent+offset,
          axis=false, legend=false, title="acc: $(Int(round(accRate*100)))%",
          c=:grey70, grid=false, titlefont=traceFont, linewidths=.5)

  end

  xmean = mean(x)
  xsd = std(x)
  q = quantile(x, [a/2, 1-a/2])

  xticks!(round.([q; xmean], digits=q_digits), axiscolor=:transparent; kw...)
  #vline!(q, c=:red, linewidths=1, line=:dot, label="95% CI", legend=:best; kw...)
  #vline!([xmean], c=:red, linewidths=2, legend=:best,
  #       label="Mean: $(round(xmean, digits=q_digits))"; kw...)
  #hline!([0], label="SD: $(round(xsd, digits=sd_digits))", 
  #       linewidth=0, bgcolor_legend=:transparent; kw...)
  vline!(q, c=:red, label="", linewidths=1, line=:dot; kw...)
  vline!([xmean], c=:red, label="", linewidths=1; kw...)
  hline!([0], linewidth=0, label="", bgcolor_legend=:transparent; kw...)

  # TODO:
  # Option for traceplot

  return img
end

function plotPosts(X::Matrix{T}; a::Float64=0.05, q_digits::Int=3, sd_digits::Int=3, cor_digits::Int=3, titles=:none, detail=false, hist2d::Bool=false, trace::Bool=true, traceFont=font(16), kw...) where {T <: Number}
  N = size(X, 2)

  if titles == :none
    titles = fill("", N)
  end

  img = plot(;layout=(N, N))
  counter = 0
  offset = 0
  for r in 1:N
    for c in 1:N
      counter += 1

      if r == c
        offset += 1
        plotPost(X[:, r], a=a, q_digits=q_digits, sd_digits=sd_digits, subplot=counter,
                 add=true, title=titles[c], trace=trace,
                 parent=counter, offset=N*N-counter+offset, traceFont=traceFont; kw...)
      elseif r < c
        if hist2d
          if detail
            histogram2d!(X[:, c], X[:,r], subplot=counter, colorbar=:none, grid=false)
          else
            histogram2d!(X[:, c], X[:,r], subplot=counter, colorbar=:none, grid=false,
                         axis=:off)
          end
        else
          plot!(X[:, c], X[:,r], subplot=counter, c=:grey70, grid=false,
                axis=false, legend=false, linewidths=.5)
        end
      else # r > c
        corXrc = cor(X[:, r], X[:, c])
        annotate!([(0.5, 0.5, text("r=$(round(corXrc, digits=cor_digits))", 
                                   Int(ceil(abs(corXrc) * 5))+5, :center))],
                  subplot=counter, axis=false, grid=false)
      end
    end
  end

  return img
end

# save by savefig
end
