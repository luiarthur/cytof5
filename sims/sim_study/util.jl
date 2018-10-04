module util
using Distributions, RCall
import Cytof5

# Import R libraries
# TODO: Remake rcommon and cytof3 for this.
R"require(rcommon)";
R"require(cytof3)";

# Import R plotting functions
plot = R"plot";
ari = R"cytof3::ari";
rgba = R"cytof3::rgba";
density = R"density";
lines = R"lines";
plotPost = R"rcommon::plotPost";
plotPosts = R"rcommon::plotPosts";
myImage = R"cytof3::my.image";
plotPdf = R"pdf";
plotPng = R"png";
devOff = R"dev.off";
blueToRed = R"cytof3::blueToRed";
greys = R"cytof3::greys";
plot_dat = R"cytof3::plot_dat";
yZ_inspect = R"cytof3::yZ_inspect";
yZ_inspect = R"cytof3::yZ_inspect";
abline = R"abline";
addErrbar = R"rcommon::add.errbar";
hist = R"hist";


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

R"""
plotPostProbMiss <- function(pmiss_mean, pmiss_lower, pmiss_upper, y_seq, i, ...) {
  plot(y_seq, pmiss_mean, xlab="y", ylab="prob miss", lwd=2, col="steelblue",
       type="l", fg="grey", ...)
  color.btwn(y_seq, pmiss_lower, pmiss_upper, from=min(y_seq), to=max(y_seq),
             rgba("blue", .3))
}
"""

plotPostProbMiss = R"plotPostProbMiss"

end # util
