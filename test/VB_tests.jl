#= If running in console:
using Test
=#
using Cytof5, Flux, Distributions, RCall
import Random
using BSON
@rimport graphics
@rimport grDevices

@testset "Variational Inference" begin
  println("Test Variational Inference...")
  # Set random seed for generating data
  Random.seed!(2)

  # Generate Data
  L = Dict(0=>3, 1=>3)
  J = 20
  K = 5

  println("Simulate Data...")
  N = [8, 1, 2] * 5000

  I = length(N)
  mus=Dict(0 => -[1.0, 2.3, 3.5], 
           1 => +[1.0, 2.0, 3.0])
  a_W=rand(K)*10
  a_eta=Dict(z => rand(L[z])*10 for z in 0:1)
  Z = Cytof5.Model.genSimpleZ(J, K)
  @time dat = Cytof5.Model.genData(J=J, N=N, K=K, L=L, Z=Z,
                                   beta=[-9.2, -2.3],
                                   sig2=[0.2, 0.1, 0.3],
                                   mus=mus,
                                   a_W=a_W,
                                   a_eta=a_eta,
                                   sortLambda=false,
                                   propMissingScale=0.7,
                                   eps=fill(.005, I))

  tau = .0001 # FIXME: why can't I do .001?
  use_stickbreak = false
  noisy_var = 10.0
  K_MCMC = 10
  L_MCMC = Dict(false=>5, true=>3)
  priors = Cytof5.VB.Priors(K=K_MCMC, L=L_MCMC, use_stickbreak=use_stickbreak)
  mc = Cytof5.Model.defaultConstants(Cytof5.Model.Data(dat[:y]),
                                     K_MCMC, Dict{Int64,Int64}(L_MCMC),
                                     yQuantiles=[0.0, 0.25, 0.5],
                                     pBounds=[.05, .8, .05])

  beta = [mc.beta[:, i] for i in 1:I]
  c = Cytof5.VB.Constants(I, N, J, K_MCMC, L_MCMC, tau, beta, use_stickbreak,
                          noisy_var, priors)
  y = dat[:y]

  println("fit model...")
  # niters = 20000
  # batchsize = 2000
  # to test compilation
  niters = 100
  batchsize = 200
  opt = ADAM(1e-2)
  out = Cytof5.VB.fit(y=y, niters=niters, batchsize=batchsize, c=c, opt=opt,
                      seed=0, nsave=30)

  # Save results
  mkpath("result/vb")
  @time BSON.bson("result/vb/vb-out.bson", out)

  #= Load results
  using Cytof5, Flux, Distributions, BSON
  =#
  out = BSON.load("result/vb/vb-out.bson")
  samples = [Cytof5.VB.rsample(out[:state])[2] for b in 1:100]
  Z = [reshape(s.v, 1, length(s.v)) .> s.H for s in samples]

  grDevices.pdf("result/vb/results.pdf")
    graphics.plot(out[:metrics][:elbo] / sum(out[:c].N),
                  typ="l", xlab="iterations", ylab="elbo");
    graphics.image(mean(Z)' .> .5)
  grDevices.dev_off()

  @test true
end
