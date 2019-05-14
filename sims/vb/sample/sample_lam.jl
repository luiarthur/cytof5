using Cytof5
using Distributions
using Flux, Flux.Tracker
include("util.jl")
include("logprob_lam.jl")

function sample_lam(state::Cytof5.VB.StateMP, y::Vector{Matrix{Float64}}, c::Cytof5.VB.Constants)
  _, s, ys, _ = Cytof5.VB.rsample(state, y, c)
  lp_lam= logprob_lam(data(s), Tracker.data.(ys), c)
  lam = [[Cytof5.MCMC.wsample_logprob(lp_lam[i][n, :])
          for n in 1:c.N[i]] for i in 1:c.I]

  # Relabel noisy class as 0
  for i in 1:c.I
    for n in 1:c.N[i]
      if lam[i][n] > c.K
        lam[i][n] = 0
      end
    end
  end

  return Vector{Int8}.(lam)
end


VV = Vector{Vector{I}} where I
VVV = Vector{VV{I}}  where I

function lam_f(lam_samps::VVV{I}, i::Integer, n::Integer, f::Function) where {I <: Integer}
  B = length(lam_samps)
  return f(lam_samps[b][i][n] for b in 1:B)
end

function lam_f(lam_samps::VVV{I}, f::Function) where {I <: Integer}
  I = length(lam_samps[1])
  N = length.(lam_samps[1])
  B = length(lam_samps)

  return [[lam_f(lam_samps, i, n, f) for n in 1:N[i]] for i in 1:I]
end

#= TEST -------------------------------------------------------
# read data
using BSON
output_path = "../results/vb-sim-paper/test/0/output.bson"
out = BSON.load(output_path)
simdat = out[:simdat]
simdat[:y] = Matrix{Float64}.(simdat[:y])

c = out[:c]
elbo = out[:metrics][:elbo] / sum(c.N)
state = out[:state]
state_hist = out[:state_hist]
metrics = out[:metrics]
m = [isnan.(yi) for yi in simdat[:y]]

# Test stuff
@time lam_samps = [sample_lam(state, simdat[:y], c) for b in 1:10]
@time lam_mode = lam_f(lam_samps, mode)
=#
