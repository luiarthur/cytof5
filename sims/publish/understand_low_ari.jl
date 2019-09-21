using Distributions
using Cytof5, Random
using JLD2, FileIO
import StatsBase

using RCall
@rimport mclust

# Directory containing CB results 
data_dir = "/scratchdata/alui2/cytof/results/cb/"

# Path to mm1 and best output
path_to_mm0_output = "$(data_dir)/best/output.jld2"
path_to_mm1_output = "$(data_dir)/mm1/output.jld2"

# Get output for miss-mech-0 and miss-mech-1
mm0 = load(path_to_mm0_output)
mm1 = load(path_to_mm1_output)
K = mm0["c"].K

# Get one of the clusterings from each
samp_idx = 1
lam0 = mm0["out"][1][samp_idx][:lam]
lam1 = mm1["out"][1][samp_idx][:lam]

# Compute ARI for any two samples
function sample_ari(mcmc1, mcmc2)
  # Number of mcmc samples
  nmcmc = length(mm0["out"][1])

  # Get a pair of mcmc samples
  i, j = StatsBase.samplepair(nmcmc)
  
  ari = mclust.adjustedRandIndex(
    mcmc1[i][:lam][1],
    mcmc2[j][:lam][1]
  )

  return Float64(ari)
end

function sample_aris(mcmc1, mcmc2; nsamps::Int)
  return [sample_ari(mcmc1, mcmc2) for _ in 1:nsamps]
end

sample_aris(mm0["out"][1], mm1["out"][1], nsamps=100)
