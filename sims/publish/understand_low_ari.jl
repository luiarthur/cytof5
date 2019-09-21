using Distributions
using Cytof5, Random
using JLD2, FileIO
import StatsBase

using RCall
@rimport mclust
@rimport base as rbase

# Directory containing CB results 
data_dir = "/scratchdata/alui2/cytof/results/cb/"

# Path to mm1 and best output
path_to_mm0_output = "$(data_dir)/best/output.jld2"
path_to_mm1_output = "$(data_dir)/mm1/output.jld2"
path_to_mm2_output = "$(data_dir)/mm2/output.jld2"

# Get output for miss-mech-0 and miss-mech-1
mm0 = load(path_to_mm0_output)
mm1 = load(path_to_mm1_output)
mm2 = load(path_to_mm2_output)
K = mm0["c"].K

# Get one of the clusterings from each
samp_idx = 1

# Compute ARI for any two samples
function sample_ari(mcmc1::T, mcmc2::T, i::Integer) where T
  # Number of mcmc samples
  nmcmc = length(mcmc1)

  # Get a pair of mcmc samples
  idx_1, idx_2 = StatsBase.samplepair(nmcmc)
  
  # Compute ARI
  ari = mclust.adjustedRandIndex(
    mcmc1[idx_1][:lam][i],
    mcmc2[idx_2][:lam][i]
  )

  return ari[1]
end

function sample_ari(mcmc1::T, mcmc2::T, i::Integer, nsamps::Integer) where T
  return [sample_ari(mcmc1, mcmc2, i) for _ in 1:nsamps]
end

function sample_ari_major(mcmc1::T, mcmc2::T, i::Integer, min_w::Float64) where T
  # Number of mcmc samples
  nmcmc = length(mcmc1)

  # Get a pair of mcmc samples
  idx_1, idx_2 = StatsBase.samplepair(nmcmc)

  # Get W for each mcmc sample
  w_1 = mcmc1[idx_1][:W][i, :]

  # Clusters to keep
  cluster_to_keep = findall(x -> x, w_1 .> min_w)

  # Observations to keep
  idx_keep = findall(cluster_label -> cluster_label in cluster_to_keep,
                     mcmc1[idx_1][:lam][i])
  
  # Compute ARI
  ari = mclust.adjustedRandIndex(
    mcmc1[idx_1][:lam][i][idx_keep],
    mcmc2[idx_2][:lam][i][idx_keep]
  )

  return ari[1]
end

function sample_ari_major(mcmc1::T, mcmc2::T, i::Integer,
                          min_w::Float64, nsamps::Integer) where T
  return [sample_ari_major(mcmc1, mcmc2, i, min_w) for _ in 1:nsamps]
end

i = 1
sample_ari(mm0["out"][1], mm0["out"][1], i)
maximum(sample_ari(mm0["out"][1], mm0["out"][1], i, 100))

mean(sample_ari_major(mm0["out"][1], mm1["out"][1], 1, .1, 100))
mean(sample_ari_major(mm0["out"][1], mm2["out"][1], 1, .1, 100))

aris_mm0 = sample_ari(mm0["out"][1], mm0["out"][1], i, 100)
aris_mm0_mm1 = sample_ari(mm0["out"][1], mm1["out"][1], i, 100)
aris_mm0_mm2 = sample_ari(mm0["out"][1], mm2["out"][1], i, 100)

mean(aris_mm0)
mean(aris_mm0_mm1)
mean(aris_mm0_mm2)

mean(aris_mm0_mm1 ./ aris_mm0)
mean(aris_mm0_mm2 ./ aris_mm0)
