module util
using Distributions, RCall
import Cytof5

# Import R libraries
# TODO: Remake rcommon and cytof3 for this.
R"require(rcommon)";
R"require(cytof3)";

# Import R plotting functions
plot = R"plot";
boxplot = R"boxplot";
ari = R"cytof3::ari";
rgba = R"rcommon::rgba";
density = R"density";
lines = R"lines";
plotPost = R"rcommon::plotPost";
plotPosts = R"rcommon::plotPosts";
myImage = R"cytof3::my.image";
plotPdf = R"pdf";
# plotPng = R"png";
R"""
plotPng = function(fname, s=10, w=480, h=480, ps=12, ...) {
  png(fname, w=w*s, h=h*s, pointsize=ps*s, ...)
}
""";
plotPng = R"plotPng";
devOff = R"dev.off";
blueToRed = R"cytof3::blueToRed";
greys = R"cytof3::greys";
plot_dat = R"cytof3::plot_dat";
yZ_inspect = R"cytof3::yZ_inspect";
yZ= R"cytof3::yZ";
abline = R"abline";
addErrbar = R"rcommon::add.errbar";
hist = R"hist";
colorBtwn = R"color.btwn";
myQQ = R"my.qqplot";
# addCut = R"add.cut";
addCut = R"function(clus, s=1) abline(h=cumsum(table(clus)) + .5, lwd=3*s, col='yellow')";
addGridLines = R"function(Z, s=1) abline(v=1:NCOL(Z) + .5, h=1:NROW(Z) + .5, col='grey', lwd=1*s)";


function quantile_vm(xs, p)
  I, J = size(xs[1])
  for x in xs
    @assert size(x) == (I, J)
  end
  return [quantile([x[i, j] for x in xs], p) for i in 1:I, j in 1:J]
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


# For Posterior Predictive QQ
function y_obs_range(y)
  I = length(y)
  y_vec = vcat([vec(y[i]) for i in 1:I]...)
  y_obs = filter(yy -> !isnan(yy), y_vec)
  return quantile(y_obs, [0, 1])
end

function idx_observed_ij(y, i::Int, j::Int)
  return findall(yij -> !isnan(yij), y[i][:, j])
end

function get_mu_observed(i::Int, j::Int, y, o) 
  z_ij = o[:Z][j, o[:lam][i]]
  eta_ij = [o[:eta][zinj][i, j, :] for zinj in z_ij]
  gam_ij = [Cytof5.MCMC.wsample(einj) for einj in eta_ij]
  idx_observed = idx_observed_ij(y, i, j)

  result = [begin
              l = gam_ij[n]
              z = z_ij[n]
              sum_delta = sum(o[:delta][z][1:l])
              z == 0 ? -sum_delta : sum_delta
            end for n in idx_observed]
  return result
end

function gen_post_pred(i, j, y, out)
  y_pp = [rand.(Normal.(get_mu_observed(i, j, y, o), sqrt(o[:sig2][i])))
          for o in out[1]]
  return hcat(y_pp...)
end

function qq_yobs_postpred(y, i::Int, j::Int, out)

  y_obs = y[i][idx_observed_ij(y, i, j), j]
  y_pp = gen_post_pred(i, j, y, out)

  return y_obs, y_pp
end

function reorder_lami(ord, lami)
  lami_new = fill(NaN, length(lami))
  K = length(ord)
  for k in 1:K
    lami_new[lami .== ord[k]] .= k
  end
  return lami_new
end

function get_common_celltypes(Wi; thresh=.9, filter_by_min_presence=false)
  ord = sortperm(Wi, rev=true)
  cumsum_Wi = cumsum(Wi[ord])

  if filter_by_min_presence
    # filter celltypes by min threshold
    return ord[Wi[ord] .> thresh]
  else
    # filter celltypes by a cumulative presence
    k = minimum(findall(cumsum_Wi .> thresh))
    return ord[1:k]
  end
end

end # util
