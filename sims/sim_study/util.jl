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
colorBtwn = R"color.btwn";
myQQ = R"my.qqplot";


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

  Amean = mean(As)
  mse = [ mean((A - Amean) .^ 2) for A in As]

  return argmin(mse)
end

function plotProbMiss(beta, i; xlim=[-10, 5],
                      ygrid=range(xlim[1], stop=xlim[2], length=300))

  p = [Cytof5.Model.prob_miss(yy, beta[:, i]) for yy in ygrid]
  plot(ygrid, p, main="i: $i", typ="l", xlab="y", ylab="prob of missing",
       xlim=xlim)
end

end # util
