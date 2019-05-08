using Cytof5
using Flux, Flux.Tracker
using Distributions
import Dates, Random
include("../../sims/cb/PreProcess.jl")

using JLD2, FileIO
function loadSingleObj(objPath)
  data = load(objPath)
  return data[collect(keys(data))[1]]
end

Random.seed!(2)

include("VB.jl")

println("test ModelParam")
@time s = VB.ADVI.ModelParam("unit");
@time v = VB.ADVI.ModelParam(3, "unit");
@time a = VB.ADVI.ModelParam((3, 5), "unit");

@time VB.ADVI.vp(s);
@time VB.ADVI.vp(v);
@time VB.ADVI.vp(a);

@time VB.ADVI.rsample(s);
@time VB.ADVI.rsample(v);
@time VB.ADVI.rsample(a);

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
  priors = VB.Priors(K_MCMC, L_MCMC, use_stickbreak=use_stickbreak)
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
  priors = VB.Priors(mc.K, L, use_stickbreak=use_stickbreak)
  c = VB.Constants(cbData.I, cbData.N, cbData.J, mc.K, L,
                   tau, beta, use_stickbreak, noisy_var, priors)
end

println("test state assignment")
if SIMULATE_DATA
  Random.seed!(1)
else
  Random.seed!(0)
end
state = VB.State(c)
metrics = Dict{Symbol, Vector{Float64}}()
for m in (:ll, :lp, :lq, :elbo) metrics[m] = Float64[] end
loss(y) = -VB.compute_elbo(state, y, c, metrics) / sum(c.N)
ps = VB.ADVI.vparams(state)
ShowTime() = Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS")

# TRAIN
println("training...")
opt = ADAM(1e-2)
minibatch_size = 2000
niters = 10000
state_hist = typeof(state)[]
for t in 1:niters
  idx = [if 0 < minibatch_size < c.N[i] 
           Distributions.sample(1:c.N[i], minibatch_size, replace=false)
         else
           1:c.N[i]
         end for i in 1:c.I]
  y_mini = [y[i][idx[i], :] for i in 1:c.I]

  # Flux.train!(loss, ps, [(y_mini, )], opt)
  gs = Tracker.gradient(() -> loss(y_mini), ps)
  Flux.Tracker.update!(opt, ps, gs)

  if t % 10 == 0
    m = ["$(key): $(round(metrics[key][end] / sum(c.N), digits=3))"
         for key in keys(metrics)]
    println("$(ShowTime()) | $(t)/$(niters) | $(join(m, " | "))")
    append!(state_hist, [deepcopy(state)])
  end
end


# POST PROCESS
using RCall
@rlibrary rcommon
@rlibrary cytof3

println("test rsample of state")
@time realp, tranp, yout, log_qy = VB.rsample(state, y, c);
# samples = [VB.rsample(s, y, c)[2] for s in state_hist[1:4:end]]
samples = [VB.rsample(s, y, c)[2] for s in state_hist]

R"plot"(metrics[:elbo][5:end]/sum(c.N), xlab="iter", ylab="elbo", typ="l")

v = hcat([s.v for s in samples]...).data
v = reshape(v, 1, c.K, length(samples))
H = cat([s.H for s in samples]..., dims=3).data
Z = Int.(v .- H .> 0)
my_image(reshape(mean(Z, dims=3), c.J, c.K))


sig2 = hcat([s.sig2 for s in samples]...).data
R"plot"(sig2[1,:], typ="l", xlab="", ylab="")
R"lines"(sig2[2,:])
R"lines"(sig2[3,:])


delta0 = hcat([s.delta0 for s in samples]...).data
R"plot"(delta0[1,:], typ="l", xlab="", ylab="")
