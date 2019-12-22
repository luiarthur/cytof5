println("pid: $(getpid())")
println("Threads: $(Threads.nthreads())")
flush(stdout)

using Revise
using Cytof5
using Random
using Distributions
using BSON

include("simulatedata.jl")

# Parse arguments
if length(ARGS) == 0
  RESULTS_DIR = "results/test-test/"
  REPFAMDISTSCALE = 10
  KMCMC = 5
  Z_idx = 3
  SEED = 0
else
  RESULTS_DIR = ARGS[1]
  REPFAMDISTSCALE = parse(Float64, ARGS[2])
  KMCMC = parse(Int, ARGS[3])
  Z_idx = parse(Int, ARGS[4])
  SEED = parse(Int, ARGS[5])
end
mkpath(RESULTS_DIR)
USE_REPULSIVE = REPFAMDISTSCALE > 0
# USE_REPULSIVE = true  # use the repulsive joint updates even when scale=0
Z = Zs[Z_idx]

# Random.seed!(0)  # before test-sims-5
Random.seed!(SEED)  # after test-sims-5

println("CONFIG:")
println("    - RESULTS_DIR: $(RESULTS_DIR)")
println("    - REPFAMDISTSCALE: $(REPFAMDISTSCALE)")
println("    - KMCMC: $(KMCMC)")
println("    - Z_idx: $(Z_idx)")
println("    - USE_REPULSIVE: $(USE_REPULSIVE)")
println("    - SEED: $(SEED)")
flush(stdout)

function sim_z_generator(phi)::Function
  # larger phi -> higher similarity -> higher penalty
  # smaller phi -> lower similarity -> lower penalty
  # phi = 0 -> regular IBP
  return (z1::Vector{Bool}, z2::Vector{Bool}) -> begin
    exp(-sum(abs.(z1 - z2)) / phi)
  end
end

sim_z = sim_z_generator(REPFAMDISTSCALE)

function init_state_const_data(simdat; K, L)
  deltaz_prior = TruncatedNormal(1.0, 0.1, 0.0, Inf)
  # deltaz_prior = TruncatedNormal(1.0, 0.3, 0.5, Inf)
  # deltaz_prior = TruncatedNormal(1.0, 1e-10, 0.5, Inf)
  # deltaz_prior = TruncatedNormal(0.0, 1.0, 0.75, Inf)
  # deltaz_prior = TruncatedNormal(0.0, 1.0, 0.9, Inf)
  # deltaz_prior = TruncatedNormal(0.0, 1.0, 1.0, Inf)
  # deltaz_prior = TruncatedNormal(0.0, 0.3, 0.5, Inf)
  # deltaz_prior = TruncatedNormal(1.0, 0.0001, 0.0, Inf)
  # deltaz_prior = TruncatedNormal(1.0, 0.0001, 0.5, Inf)
  d = Cytof5.Model.Data(simdat[:y])
  c = Cytof5.Model.defaultConstants(d, K, L,
                                    tau0=1.0, tau1=1.0,
                                    # sig2_prior=InverseGamma(3, 2),
                                    sig2_prior=InverseGamma(11, 5),  # TODO
                                    delta0_prior=deltaz_prior,
                                    delta1_prior=deltaz_prior,
                                    alpha_prior=Gamma(0.1, 10.0),
                                    yQuantiles=[.0, .25, .5], 
                                    pBounds=[.05, .8, .05],
                                    probFlip_Z=1.0,
                                    similarity_Z=sim_z)
  # s = Cytof5.Model.genInitialState(c, d)
  s = Cytof5.Model.smartInit(c, d)
  t = Cytof5.Model.Tuners(d.y, c.K)
  X = Cytof5.Model.eye(Float64, d.I)

  cfs = Cytof5.Model.ConstantsFS(c)
  dfs = Cytof5.Model.DataFS(d, X)
  sfs = Cytof5.Model.StateFS{Float64}(s, dfs)
  tfs = Cytof5.Model.TunersFS(t, s, X)

  return Dict(:dfs => dfs, :cfs => cfs, :sfs => sfs, :tfs => tfs,
              :simdat => simdat)
end

# Simulate data
# NOTE: Used before test-sims-5
# @time simdat = simulatedata1(Z=Z, seed=0, propmissingscale=.6, sortLambda=true);

# NOTE: Used on and after test-sims-5
@time simdat = simulatedata1(Z=Z,
                             W=Matrix(hcat([[.7, 0, .1, .1, .1],
                                            [.4, .1, .3, .1, .1]]...)'),
                             sig2=[.5, .5],
                             seed=SEED, propmissingscale=.6, sortLambda=true);

#= Sanity check
using PyPlot
const plt = PyPlot.plt
J = size(simdat[:Z], 1)
plt.imshow(simdat[:y][2], aspect="auto", cmap=plot_yz.blue2red.cm(7), 
           vmin=VLIM[1], vmax=VLIM[2])
plt.xticks(pyrange(J), 1:J)
plt.colorbar()
=#

# Parameters to monitor
monitor1 = [:theta__Z, :theta__v, :theta__alpha,
            :omega, :r, :theta__lam, :W_star, :theta__eta,
            :theta__W, :theta__delta, :theta__sig2]
monitor2 = [:theta__y_imputed, :theta__gam]

# MCMC Specs
nsamps_to_thin(nsamps::Int, nmcmc::Int) = max(1, div(nmcmc, nsamps))
NSAMPS = 3000  # Number of samples
THIN_SAMPS = 2  # Factor to thin the primary parameters
MCMC_ITER = NSAMPS * THIN_SAMPS  # Number of MCMC iterations
# LPML / DIC are computed based on `MCMC_ITER` samples
NBURN = 3000  # burn-in time

# Configurations: priors, initial state, data, etc.
config = init_state_const_data(simdat, K=KMCMC, L=Dict(0 => 2, 1 => 2))

# Print constants
Cytof5.Model.printConstants(config[:cfs])
flush(stdout)

# Fit model
@time out = Cytof5.Model.fit_fs!(config[:sfs], config[:cfs], config[:dfs],
                                 tuners=config[:tfs], 
                                 nmcmc=MCMC_ITER,
                                 nburn=NBURN,
                                 thins=[THIN_SAMPS, nsamps_to_thin(10, MCMC_ITER)],
                                 monitors=[monitor1, monitor2],
                                 computedden=true,
                                 thin_dden=nsamps_to_thin(200, MCMC_ITER),
                                 printFreq=10, time_updates=false,
                                 computeDIC=true, computeLPML=true,
                                 use_repulsive=USE_REPULSIVE,
                                 Z_thin=1,
                                 flushOutput=true)

# Dump output
BSON.bson("$(RESULTS_DIR)/output.bson", out)
BSON.bson("$(RESULTS_DIR)/simdat.bson", Dict(:simdat => config[:simdat]))

# TODO: send to s3?

println("Completed!")
