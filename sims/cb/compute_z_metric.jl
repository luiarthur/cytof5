using Cytof5, RCall
import Cytof5.Util.similarity_FM
@rimport graphics as plt
@rimport grDevices as dev

# Get path to results
if length(ARGS) == 0
  PATH_TO_RESULTS = "results/cb-paper/"
else
  PATH_TO_RESULTS = ARGS[1]
end
results_dirs = filter(d -> occursin("K_MCMC", d), readdir(PATH_TO_RESULTS))

struct Estimate
  Z::Vector{Matrix{Int}}
  W::Vector{Vector{Float64}}
  I::Int
  J::Int
  K::Int
end

function Estimate(Z::Vector{Matrix{Int}}, W::Vector{Vector{Float64}})
  @assert length(Z) == length(W)
  I = length(Z)
  J, K = size(Z[1])
  for i in 1:I
    @assert size(Z[i], 2) == length(W[i])
    @assert size(Z[i], 1) == J
  end
  return Estimate(Z, W, I, J, K)
end

function read_Zhat(path_to_zhat)
  rows = Vector{Int}[]
  open(path_to_zhat, "r") do f
    lines = readlines(f)
    for line in lines
      row = parse.(Int, split(line, ","))
      append!(rows, [row])
    end
  end
  Z = Matrix(hcat(rows...)')
  return Z
end

function read_What(path_to_what)
  what = nothing
  open(path_to_what, "r") do f
    lines = readlines(f)
    what = parse.(Float64, lines)
  end
  return what
end

function get_est(d)
  imgdir = "$(PATH_TO_RESULTS)/$(d)/img/"
  Z_hat_files = filter(d -> occursin(r"Z\d+_hat\.txt", d), readdir(imgdir))
  W_hat_files = filter(d -> occursin(r"W_\d+_hat\.txt", d), readdir(imgdir))
  I = length(Z_hat_files)
  @assert I == length(W_hat_files)

  Z_hats = [read_Zhat("$imgdir/$d") for d in Z_hat_files]
  W_hats = [read_What("$imgdir/$d") for d in W_hat_files]

  Estimate(Z_hats, W_hats)
end


# Read estimates
estimates = [get_est(d) for d in results_dirs]
ord = sortperm(getfield.(estimates, :K))
estimates = estimates[ord]
Ks = [est.K for est in estimates[2:end]]

z_metric = Float64[]
for r in 2:length(estimates)
  W = Matrix(hcat(estimates[r].W...)')
  zm = similarity_FM(estimates[r].Z[1], W, estimates[r-1].Z[1])
  append!(z_metric, zm)
end

dev.pdf("$(PATH_TO_RESULTS)/metrics/L0_MCMC5/z-metric.pdf")
plt.plot(Ks, z_metric, typ="o", xlab="K-MCMC", ylab="Z-metric", xaxt="n")
plt.axis(1, at=Ks, label=Ks)
dev.dev_off()

z_metric_scaled = Float64[]
for r in 2:length(estimates)
  W = Matrix(hcat(estimates[r].W...)')
  zm = similarity_FM(estimates[r].Z[1], W, estimates[r-1].Z[1], z_diff=Cytof5.Util.meanabsdiff)
  append!(z_metric_scaled, zm)
end

dev.pdf("$(PATH_TO_RESULTS)/metrics/L0_MCMC5/z-metric-scaled.pdf")
plt.plot(Ks, z_metric_scaled, typ="o", xlab="K-MCMC", ylab="Z-metric", xaxt="n")
plt.axis(1, at=Ks, label=Ks)
dev.dev_off()

