module CytofImg

import Cytof5
using Plots
pyplot()
using StatPlots
using Distributions # std, mean, quantile
using RCall # TODO: GET RID OF THIS DEPENDENCY!

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
  # FIXME: doesn't work for Z having either 1 row or 1 column!
  J, K = size(Z)
  if J < 2 || K < 2
    println("heatmap is not supported for single row / column matrices")
    return
  end

  img = heatmap(Z, c=:Greys, legend=:none, border=true, bordercolor=:lightgrey; kw...)
  if J > 1
    hline!((2:J) .- .5, c=:lightgrey)
  end
  if K > 1
    vline!((2:K) .- .5, c=:lightgrey)
  end

  return img
end

function plotY_heatmap(y::Matrix{T}; clim=(-5, 5), c=:pu_or) where {T <: Number}
  img = heatmap(y, c=c, clim=clim, background_color_inside=:black, bordercolor=:transparent);
  return img
end

function plotYZ(y::Matrix{V}, Z::Matrix{T}; ycolor=:balance, clim=(-5, 5), heights=[.7, .3], kw...) where {T <: Number, V <: Number}
  # TODO: change proportions
  heatY = plotY_heatmap(y, c=ycolor, clim=clim)
  heatZ = plotZ(Matrix(Z'))
  img = plot(heatY, heatZ, layout=grid(2, 1, heights=heights); kw...);
  return img
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
                  trace::Bool=true, useDensity::Bool=true, showAccRate::Bool=true,
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
		traceTitle = showAccRate ? "acc: $(Int(round(accRate*100)))%" : ""
    #                                x    y   W   H
    plot!(x, inset_subplots=(parent, bbox(0, .1, .3, .3, :top, :right)), subplot=parent+offset,
          axis=false, legend=false, title=traceTitle,
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

function plotPosts(X::Matrix{T}; a::Float64=0.05, q_digits::Int=3,
                   sd_digits::Int=3, cor_digits::Int=3, titles=:none,
                   detail=false, hist2d::Bool=false, trace::Bool=true,
                   traceFont=font(16), kw...) where {T <: Number}
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
        plotPost(X[:, r], a=a, q_digits=q_digits, sd_digits=sd_digits,
                 subplot=counter, add=true, title=titles[c], trace=trace,
                 parent=counter, offset=N*N-counter+offset,
                 traceFont=traceFont; kw...)
      elseif r < c
        if hist2d
          if detail
            histogram2d!(X[:, c], X[:,r], subplot=counter, colorbar=:none, grid=false)
          else
            histogram2d!(X[:, c], X[:,r], subplot=counter, colorbar=:none, grid=false,
                         axis=false)
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

function postProbMiss(b0, b1, i::Int;
                      y::Vector{Float64}=collect(-10:.1:10),
                      credibility::Float64=.95)

  N, I = size(b0)
  @assert size(b0) == size(b1)
  
  len_y = length(y)
  alpha = 1 - credibility
  p_lower = alpha / 2
  p_upper = 1 - alpha / 2

  pmiss = hcat([Cytof5.Model.prob_miss.(yi, b0[:,i], b1[:,i]) for yi in y]...)
  pmiss_mean = vec(mean(pmiss, dims=1))
  pmiss_lower = [ quantile(pmiss[:, col], p_lower) for col in 1:len_y ]
  pmiss_upper = [ quantile(pmiss[:, col], p_upper) for col in 1:len_y ]

  return (pmiss_mean, pmiss_lower, pmiss_upper, y)
end

function pairwiseAlloc(Z, W, i)
  J, K = size(Z)
  A = zeros(J, J)
  for r in 1:J
    for c in 1:J
      for k in 1:K
        if Z[r, k] == 1 && Z[c, k] == 1
          A[r, c] += W[i, k]
          A[c, r] += W[i, k]
        end
      end
    end
  end

  return A
end # pairwiseAlloc

function estimate_ZWi_index(monitor, i)
  As = [pairwiseAlloc(m[:Z], m[:W], i) for m in monitor]

  Amean = mean(As)
  mse = [ mean((A - Amean) .^ 2) for A in As]

  return argmin(mse)
end

function countmap(x::Vector{T}) where {T <: Number}
  out = Dict{Int, T}()
  for xi in x
    if xi in keys(out)
      out[xi] += 1
    else
      out[xi] = 1
    end
  end
  return out
end

function reorder_lam_i(lam_i::Vector{T}, W_i::Vector{V}) where {T <: Number, V <: Number}
  N = length(lam_i)
  new_lam_i = Vector{Int}(undef, N)
  K = length(W_i)
  ord = sortperm(W_i)

  for k in 1:K
    new_lam_i[lam_i .== ord[k]] .= k
  end

  return new_lam_i
end

# TODO: Implement the threshold
function yZ_inspect(monitor, y::Vector{Matrix{Float64}},
                    i::Int; thresh=0.7, ycolor=:balance,
                    addLines::Bool=true, marker_names = missing,
                    clim=(-5,5), W_digits=1, kw...)
  idx_best = estimate_ZWi_index(monitor, i)
  Zi = monitor[idx_best][:Z]
  Wi = monitor[idx_best][:W][i, :]
  lami = monitor[idx_best][:lam][i]
  new_lami = reorder_lam_i(lami, Wi)
  ord = sortperm(new_lami)
  K = length(Wi)
  J = size(y[i], 2)

  W_order = sortperm(Wi)

  cumProb = 0.0; k = K
  for w in reverse(Wi[W_order])
    cumProb += w
    if cumProb > thresh
      break
    end
    k -= 1
  end
  W_order = W_order[k:end]

  plotYZ(y[i][ord, :], Zi[:, W_order], clim=clim, ycolor=ycolor)

  # TODO: cell types on left, percentage on right
  W_val = map(x->"$(x)%", round.(Wi[W_order] * 100, digits=W_digits))
  #plot!(yticks=(1:K, W_order), subplot=2)
  plot!(yticks=(1:length(W_order), W_val), subplot=2, ymirror=true)

  # add marker names if provided
  if !ismissing(marker_names)
    plot!(xticks=(1:J, marker_names), subplot=1, rotation=90)
  end

  if addLines
    groups = cumsum(sort(collect(values(countmap(new_lami)))))
    hline!(groups, c=:yellow, subplot=1, linewidths=1, legend=false)
  end
end

function postSummary(x::Matrix{T}; alpha::Float64=.05) where {T <: Number}
  K = size(x, 2)
  xmean = mean(x, dims=1)
  xsd = std(x, dims=1)
  xqLower = [ quantile(x[:, k], alpha / 2) for k in 1:K ]
  xqLower = reshape(xqLower, 1, K)
  xqUpper = [ quantile(x[:, k], 1- alpha / 2) for k in 1:K ]
  xqUpper = reshape(xqUpper, 1, K)
  xmedian = median(x, dims=1)

  summaryMatrix = Matrix([xmean; xsd; xqLower; xmedian; xqUpper]')
  println("mean   sd  lower  upper  median")
  return summaryMatrix
end

# TODO: GET RID OF THIS!
R"""
ari <- function(x, y) {
  x = as.vector(x)
  y = as.vector(y)
  n = length(x)
  stopifnot(n == length(y))
  tab = table(x, y)
  index = sum(choose(tab, 2))
  rowSumsChoose2.sum = sum(choose(rowSums(tab), 2))
  colSumsChoose2.sum = sum(choose(colSums(tab), 2))
  expected.index = rowSumsChoose2.sum * colSumsChoose2.sum/choose(n,
      2)
  max.index = mean(c(rowSumsChoose2.sum, colSumsChoose2.sum))
  (index - expected.index)/(max.index - expected.index)
}
"""
ari = R"ari"

function confusion(x::Vector{T}, y::Vector{T}) where {T <: Number}
  @assert length(y) == length(x)
  n = length(x)
  confusionMat = zeros(Int, n, n)
  for i in 1:n
    for j in 1:n
      if x[i] == y[j]
        confusionMat[i, j] += 1
      end
    end
  end
  return confusionMat
end

#=
function ari(x::Vector{T}, y::Vector{T}) where {T <: Number}
  @assert length(y) == length(x)
  n = length(x)
  tab = confusion(x, y)
  # TODO: https://en.wikipedia.org/wiki/Rand_index
end
=#



end
