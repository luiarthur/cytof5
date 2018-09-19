module util
using Distributions, RCall
import Cytof5

R"require(rcommon)";
R"require(cytof3)";

plot = R"plot";
plotPost = R"rcommon::plotPost"
plotPosts = R"rcommon::plotPosts"
colorBtwn = R"rcommon::color.btwn"
myImage = R"cytof3::my.image"
plotPdf = R"pdf"
devOff = R"dev.off"
blueToRed = R"blueToRed"
greys = R"cytof3::greys"
rgba = R"rcommon::rgba"
abline = R"abline"

function matMean(X)
  X_mean = zeros(size(X[1]))
  N = length(X)
  for x in X
    X_mean += x / N
  end
  return X_mean
end

function getPosterior(sym::Symbol, monitor)
  return [ m[sym] for m in monitor]
end

nrow(X) = size(X, 1)
ncol(X) = size(X, 2)

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

  Amean = matMean(As)
  mse = [ mean((A - Amean) .^ 2) for A in As]

  return argmin(mse)
end

# TODO: Test
#function fy(lami)
#  abline(h=cumsum(R"table($lami)") .+ .5, lwd=3, col="yellow", lty=1)
#end
#function fZ(Z)
#  J, K = size(Z)
#  abline(v=1:K .+ .5, h=1:J .+ .5, col="grey")
#end
#function yZ_inspect(out, y, zlim, i, thresh=0.7, col=blueToRed(7),
#                    propLowerPanel=0.3, isPostPred=false, decimalsW=1, 
#                    naColor="transparent",
#                    fy=fy, fZ=fZ, main="", addL=true)
#  """
#  fy: Function to execute after making y image
#  fZ: Function to execute after making Z image
#  """
#  # TODO
#end

function plotPostProbMiss(b0, b1, i::Int;
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
  plot(y, pmiss_mean, xlab="y", ylab="prob miss", lwd=2, col="steelblue",
       typ="l", fg="grey")
  colorBtwn(y, pmiss_lower, pmiss_upper, from=minimum(y), to=maximum(y),
            rgba("blue", .3))
end

end # util
