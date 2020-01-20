module Sim

using Revise
using Cytof5
using Random
using Distributions
using BSON

include("../simulatedata.jl")

function simfunc(settings::Dict{Symbol, Any})
  println("pid: $(getpid())")
  println("Threads: $(Threads.nthreads())")

  println("settings:")
  println(settings)

  function sim_z_generator(phi)::Function
    # larger phi -> higher similarity -> higher penalty
    # smaller phi -> lower similarity -> lower penalty
    # phi = 0 -> regular IBP
    return (z1::Vector{Bool}, z2::Vector{Bool}) -> begin
      exp(-sum(abs.(z1 - z2)) / phi)
    end
  end

  sim_z = sim_z_generator(settings[:repfam_dist_scale])

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

  @time simdat = simulatedata1(Z=Z,
                               # N=[300, 300],  # for all but test-sims-5-8
                               # N =[500, 500],  # for test-sims-5-8
                               N=[1000, 1000],  # test-sims-6-1
                               W=Matrix(hcat([[.7, 0, .1, .1, .1],
                                              [.4, .1, .3, .1, .1]]...)'),
                               sig2=[.5, .5],
                               seed=SEED, propmissingscale=.6, sortLambda=true);


end  # simfunc
end  # module Sim
