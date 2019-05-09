using Cytof5
import Random
using Flux
include("../../../sims/cb/PreProcess.jl")

using JLD2, FileIO
function loadSingleObj(objPath)
  data = load(objPath)
  return data[collect(keys(data))[1]]
end

Random.seed!(2)

include("../VB.jl")

tau = .005 # FIXME: why can't I do .001?
use_stickbreak = false
noisy_var = 10.0

# SIMULATE_DATA = false
SIMULATE_DATA = true
if SIMULATE_DATA
  L = Dict(0=>3, 1=>3)
  I = 3
  J = 20
  K = 5

  println("Simulate Data")
  N = [8, 1, 2] * 500
  # @time dat = Cytof5.Model.genData(J, N, K, Dict{Int,Int}(L))
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
                                   sortLambda=false, propMissingScale=0.7,
                                   eps=fill(.005, I))

  I = length(N)
  K_MCMC = 10
  L_MCMC = Dict(false=>5, true=>3)
  priors = VB.Priors(K=K_MCMC, L=L_MCMC, use_stickbreak=use_stickbreak)
  mc = Cytof5.Model.defaultConstants(Cytof5.Model.Data(dat[:y]),
                                     K_MCMC, Dict{Int64,Int64}(L_MCMC),
                                     yQuantiles=[0.0, 0.25, 0.5], pBounds=[.05, .8, .05])
  beta = [mc.beta[:, i] for i in 1:I]
  c = VB.Constants(I, N, J, K_MCMC, L_MCMC, tau, beta, use_stickbreak, noisy_var, priors)
  y = dat[:y]
else
  cbDataPath = "../../sims/cb/data/cytof_cb_with_nan.jld2"
  y = loadSingleObj(cbDataPath)
  K_MCMC=30
  L = Dict(false=>5, true=>3)
  goodColumns, _ = PreProcess.preprocess!(y, maxNanOrNegProp=.9, maxPosProp=.9,
                                          subsample=0.0, rowThresh=-6.0)
  cbData = Cytof5.Model.Data(y)
  mc = Cytof5.Model.defaultConstants(cbData,
                                     K_MCMC, Dict{Int64,Int64}(L),
                                     yQuantiles=[0.0, 0.25, 0.5], pBounds=[.05, .8, .05])
  beta = [mc.beta[:, i] for i in 1:cbData.I]
  priors = VB.Priors(K=mc.K, L=L, use_stickbreak=use_stickbreak)
  c = VB.Constants(cbData.I, cbData.N, cbData.J, mc.K, L,
                   tau, beta, use_stickbreak, noisy_var, priors)
end

if SIMULATE_DATA
  Random.seed!(1)
else
  Random.seed!(0)
end

println("fit model...")
out = VB.fit(y=y, niters=10000, batchsize=2000, c=c, opt=ADAM(1e-2))


# POST PROCESS
using RCall
@rlibrary rcommon
@rlibrary cytof3

# Plot ELBO trace
R"plot"(metrics[:elbo]/sum(c.N), xlab="iter", ylab="elbo", typ="l")

# Number of MC samples
B = 100

println("test rsample of state")
# @time realp, tranp, yout, log_qy = VB.rsample(state, y, c);
@time trace = [VB.rsample(s, y, c)[2] for s in state_hist[1:20:end]];
sig2_trace = hcat([s.sig2.data for s in trace]...)
R"plot"(sig2_trace[1,:], typ="l", xlab="sig2", ylab="trace", col=2, lwd=2,
        ylim=[minimum(sig2_trace), maximum(sig2_trace)]);
R"lines"(sig2_trace[2,:], col=3, lwd=2);
R"lines"(sig2_trace[3,:], col=4, lwd=2);


# Draw MC samples from variational distributions
@time samples = [VB.rsample(state)[2] for i in 1:B];


v = hcat([s.v for s in samples]...).data
v = reshape(v, 1, c.K, length(samples))
H = cat([s.H for s in samples]..., dims=3).data
Z = Int.(v .- H .> 0)
my_image(reshape(mean(Z, dims=3), c.J, c.K))

sig2 = hcat([s.sig2 for s in samples]...).data
if SIMULATE_DATA
  ymin = minimum(minimum.([dat[:sig2], sig2]))
  ymax = maximum(maximum.([dat[:sig2], sig2]))
  R"boxplot"(sig2', typ="l", xlab="", ylab="", ylim=[ymin, ymax]);
  R"abline"(h=dat[:sig2], col="grey", lty=2);
else
  R"boxplot"(sig2', typ="l", xlab="", ylab="");
end

# Plot mu
delta0 = hcat([s.delta0 for s in samples]...).data
mu0 = -cumsum(delta0, dims=1)
delta1 = hcat([s.delta1 for s in samples]...).data
mu1 = cumsum(delta1, dims=1)
mu = [mu0; mu1]

R"boxplot"(mu', typ="l", xlab="", ylab="");
if SIMULATE_DATA
  R"abline"(h=dat[:mus][0], col="grey", lty=2);
  R"abline"(h=dat[:mus][1], col="grey", lty=2);
end
R"abline"(v=c.L[0] + .5, h=0, col="grey");

