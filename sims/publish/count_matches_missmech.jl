using Distributions
using Cytof5
using JLD2, FileIO
using PyPlot

# Directory containing CB results 
data_dir = "/scratchdata/alui2/cytof/results/cb/"

# Path to mm1 and best output
path_to_mm0_output = "$(data_dir)/best/output.jld2"
path_to_mm1_output = "$(data_dir)/mm1/output.jld2"
path_to_mm2_output = "$(data_dir)/mm2/output.jld2";

# Get output for miss-mech-0 and miss-mech-1
mm0 = load(path_to_mm0_output)
mm1 = load(path_to_mm1_output)
mm2 = load(path_to_mm2_output)

# Get K from output
K = mm0["c"].K

# Zs
Zs_mm0 = [m[:Z] for m in mm0["out"][1]]
Zs_mm1 = [m[:Z] for m in mm1["out"][1]]
Zs_mm2 = [m[:Z] for m in mm2["out"][1]]

# Last Z
Zhat_mm0 = Zs_mm0[end]
Zhat_mm1 = Zs_mm1[end]
Zhat_mm2 = Zs_mm2[end]

# Ws
Ws_mm0 = cat([m[:W] for m in mm0["out"][1]]..., dims=3)
Ws_mm1 = cat([m[:W] for m in mm1["out"][1]]..., dims=3)
Ws_mm2 = cat([m[:W] for m in mm2["out"][1]]..., dims=3)

What_mm0 = mean(Ws_mm0, dims=3)[:, :, 1]
What_mm1 = mean(Ws_mm1, dims=3)[:, :, 1]
What_mm2 = mean(Ws_mm2, dims=3)[:, :, 1] 


function countmatches(Z1, Z2; discrepancy=0)
  K1 = size(Z1, 2)
  K2 = size(Z2, 2)
  k1s = Int8[]
  k2s = Int8[]

  for k1 in 1:K1
    for k2 in 1:K2
      if sum(abs.(Z1[:, k1] - Z2[:, k2])) <= discrepancy
        append!(k1s, k1)
        append!(k2s, k2)
      end
    end
  end

  return unique(k1s), unique(k2s)
end

function visualize(Z1, Z2, W1, i; discrepancy=0, plot=true)
  @assert size(Z1, 2) == size(Z2, 2)
  k1s, k2s = countmatches(Z1, Z2, discrepancy=discrepancy)
  K = size(Z1, 2)
  matched_features = k1s
  unmatched_features = setdiff(1:K, k1s)

  if ndims(W1) == 3
    if plot
      PyPlot.boxplot(W1[i, [matched_features; unmatched_features], :]',
                     showfliers=false)
    end
    W1_hat = mean(W1, dims=3)[:, :, 1]
  elseif ndims(W1) == 2
    if plot
      PyPlot.scatter(1:K, W1[i, [matched_features; unmatched_features]])
    end
    W1_hat = W1 .+ 0
  else
    throw("ndims(W1) needs to be 2 or 3!")
  end

  num_matches_1 = length(unique(k1s))
  num_matches_2 = length(unique(k2s))

  if plot
    PyPlot.axvline(num_matches_1 + .5, ls="--", color="grey")
    PyPlot.axhline(0, ls="--", color="grey")
    PyPlot.xticks(1:K, [matched_features; unmatched_features])
  end

  println("number of matched features: $((num_matches_1, num_matches_2))")
  println("mean W for matched features: $(mean(W1_hat[i, matched_features]))")
  println("mean W for unmatched features: $(mean(W1_hat[i, unmatched_features]))")
  println("sum W for matched features: $(sum(W1_hat[i, matched_features]))")
  println("sum W for unmatched features: $(sum(W1_hat[i, unmatched_features]))")
end

# MM0 vs MM1
for d in [0, 1, 2, 3]
  for i in 1:3
    println("i: $(i), d: $(d)")
    visualize(Zhat_mm0, Zhat_mm1, Ws_mm0, i, discrepancy=d, plot=false)
  end
  println()
end

# MM0 vs MM2
for d in [0, 1, 2, 3]
  for i in 1:3
    println("i: $(i), d: $(d)")
    visualize(Zhat_mm0, Zhat_mm2, Ws_mm0, i, discrepancy=d, plot=false)
  end
  println()
end

