# NOTE: These libs have to be imported in
#       main process and worker processes.
import Pkg; Pkg.activate("../../")  # sims
using Cytof5, Random, Distributions, BSON

include("../../simulatedata.jl")
include("../../../Util/Util.jl")

function simfn(settings::Dict{Symbol, Any})
  println("pid: $(getpid())")
  println("Threads: $(Threads.nthreads())")
  println("pwd: $(pwd())")
  flush(stdout)

  println("settings:")
  println(settings)

  # Results dir
  results_dir = settings[:results_dir]
  mkpath(results_dir)

  function sim_z_generator(phi)::Function
    # larger phi -> higher similarity -> higher penalty
    # smaller phi -> lower similarity -> lower penalty
    # phi = 0 -> regular IBP
    return (z1::Vector{Bool}, z2::Vector{Bool}) -> begin
      exp(-sum(abs.(z1 - z2)) / phi)
    end
  end

  sim_z = sim_z_generator(settings[:repfam_dist_scale])
  use_repulsive = settings[:repfam_dist_scale] > 0

  function init_state_const_data(simdat; K, L)
    deltaz_prior = TruncatedNormal(1.0, 0.1, 0.0, Inf)
    d = Cytof5.Model.Data(simdat[:y])
    c = Cytof5.Model.defaultConstants(d, K, L,
                                      tau0=1.0, tau1=1.0,
                                      sig2_prior=InverseGamma(11, 5),
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

  @time simdat = simulatedata1(Z=Zs[3],
                               N=[300, 300],
                               W=Matrix(hcat([[.7, 0, .1, .1, .1],
                                              [.4, .1, .3, .1, .1]]...)'),
                               sig2=[.5, .5],
                               seed=settings[:seed_data],
                               propmissingscale=.6,
                               sortLambda=true);

  # Parameters to monitor
  monitor1 = [:theta__Z, :theta__v, :theta__alpha,
              :omega, :r, :theta__lam, :W_star, :theta__eta,
              :theta__W, :theta__delta, :theta__sig2]
  monitor2 = [:theta__y_imputed, :theta__gam]

  # MCMC Specs
  nsamps_to_thin(nsamps::Int, nmcmc::Int) = max(1, div(nmcmc, nsamps))
  NSAMPS = 20  # Number of samples
  THIN_SAMPS = 2  # Factor to thin the primary parameters
  MCMC_ITER = NSAMPS * THIN_SAMPS  # Number of MCMC iterations

  # LPML / DIC are computed based on `MCMC_ITER` samples
  NBURN = 100  # burn-in time

  # Configurations: priors, initial state, data, etc.
  config = init_state_const_data(simdat,
                                 K=settings[:Kmcmc],
                                 L=Dict(0 => 2, 1 => 2))

  # Print constants
  println("N: $(config[:dfs].data.N)")
  println("J: $(config[:dfs].data.J)")
  Cytof5.Model.printConstants(config[:cfs])
  flush(stdout)

  # Fit model
  @time out = Cytof5.Model.fit_fs!(config[:sfs], config[:cfs], config[:dfs],
                                   tuners=config[:tfs], 
                                   nmcmc=MCMC_ITER,
                                   nburn=NBURN,
                                   thins=[THIN_SAMPS,
                                          nsamps_to_thin(10, MCMC_ITER)],
                                   monitors=[monitor1, monitor2],
                                   computedden=true,
                                   thin_dden=nsamps_to_thin(200, MCMC_ITER),
                                   printFreq=10, time_updates=false,
                                   computeDIC=true, computeLPML=true,
                                   use_repulsive=use_repulsive,
                                   Z_thin=1,
                                   flushOutput=true, 
                                   seed=settings[:seed_mcmc])

  # Dump output
  BSON.bson("$(results_dir)/output.bson", out)
  BSON.bson("$(results_dir)/simdat.bson", Dict(:simdat => config[:simdat]))

  println("Completed!")
end  # simfn
