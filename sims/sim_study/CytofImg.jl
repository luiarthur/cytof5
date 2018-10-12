module CytofImg

using Plots
pyplot()
using Distributions # std, mean, quantile


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

function plotPost(x::Vector{T}; a::Float64=0.05, q_digits::Int=3, sd_digits::Int=3, add=false, trace::Bool=false, kw...) where {T <: Number}
  if add
    img = histogram!(x, label="", normed=true, grid=:off,
                     bordercolor=:white, yaxis=:off,
                     linecolor=:transparent, c=:steelblue; kw...)
  else
    img = histogram(x, label="", normed=true, grid=:off,
                    bordercolor=:white, yaxis=:off,
                    linecolor=:transparent, c=:steelblue; kw...)
  end

  if trace
    #                       x     y    W     H
    plot!(x, inset=(1, bbox(0, 0, 0.3, 0.3, :top, :right)),
          axis=:off, bg_inside=nothing; kw...)
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

function plotPosts(X::Matrix{T}; a::Float64=0.05, q_digits::Int=3, sd_digits::Int=3, cor_digits::Int=3, titles=:none, detail=false, kw...) where {T <: Number}
  N = size(X, 2)

  if titles == :none
    titles = fill("", N)
  end

  img = plot(;layout=(N, N))
  counter = 0
  for r in 1:N
    for c in 1:N
      counter += 1

      if r == c
        plotPost(X[:, r], a=a, q_digits=q_digits, sd_digits=sd_digits, subplot=counter, add=true, title=titles[c])
      elseif r < c
        if detail
          histogram2d!(X[:, c], X[:,r], subplot=counter, colorbar=:none, grid=:off)
        else
          histogram2d!(X[:, c], X[:,r], subplot=counter, colorbar=:none, grid=:off,
                       axis=:off)
        end
      else # r > c
        corXrc = cor(X[:, r], X[:, c])
        annotate!([(0.5, 0.5, text("r=$(round(corXrc, digits=cor_digits))", 
                                   Int(ceil(abs(corXrc) * 5))+5, :center))],
                  subplot=counter, axis=:off, grid=:off)
      end
    end
  end

  return img
end

# save by savefig
end
