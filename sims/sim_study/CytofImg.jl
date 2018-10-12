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

function plotPost(x::Vector{T}; a::Float64=0.05, q_digits::Int=3, sd_digits::Int=3, add=false, kw...) where {T <: Number}
  if add
    img = histogram!(x[:,1], label="", normed=true,
                     linecolor=:transparent, c=:steelblue; kw...)
  else
    img = histogram(x[:,1], label="", normed=true,
                    linecolor=:transparent, c=:steelblue; kw...)
  end

  xmean = mean(x)
  xsd = std(x)
  q = quantile(x, [a/2, 1-a/2])
  xticks!(round.(q, digits=q_digits))
  vline!(q, c=:red, linewidths=2, line=:dot, label="95% CI", legend=true)
  vline!([xmean], c=:red, linewidths=5, legend=true, label="mean")
  hline!([0], label="SD: $(round(xsd, digits=sd_digits))", linewidth=0)

  return img
end

function plotPosts(X::Matrix{T}; a::Float64=0.05, q_digits::Int=3, sd_digits::Int=3, kw...) where {T <: Number}
  N = size(X, 2)

  img = plot(;layout=(N, N))
  counter = 0
  for r in 1:N
    for c in 1:N
      counter += 1

      if r == c
        plotPost(X[:, r], a=a, q_digits=q_digits, sd_digits=sd_digits, subplot=counter, add=true)
      elseif r < c
        histogram2d!(X[:, r], X[:,c], subplot=counter)
      else # r > c
      end
    end
  end

  return img
end

# save by savefig
end
