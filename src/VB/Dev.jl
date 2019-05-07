using Cytof5
using Flux, Flux.Tracker
using Distributions
import Dates, Random

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

SIMULATE_DATA = true
if SIMULATE_DATA
  L = Dict(false=>5, true=>3)
  I = 3
  J = 20
  K = 4

  println("Simulate Data")
  N = [3, 1, 2] * 10000
  @time dat = Cytof5.Model.genData(J, N, K, Dict{Int,Int}(L))
  I = length(N)
  K_MCMC = 10
  priors = VB.Priors(K_MCMC, L, use_stickbreak=use_stickbreak)
  mc = Cytof5.Model.defaultConstants(Cytof5.Model.Data(dat[:y]),
                                     K_MCMC, Dict{Int64,Int64}(L),
                                     yQuantiles=[0.0, 0.25, 0.5], pBounds=[.05, .8, .05])
  beta = [mc.beta[:, i] for i in 1:I]
  c = VB.Constants(I, N, J, K_MCMC, L, tau, beta, use_stickbreak, noisy_var, priors)
  y = dat[:y]
else
  cbDataPath = "../../sims/cb/data/cytof_cb_with_nan.jld2"
  y = loadSingleObj(cbDataPath)
  K_MCMC=30
  L = Dict(false=>5, true=>3)
  cbData = Cytof5.Model.Data(y)
  mc = Cytof5.Model.defaultConstants(cbData, K_MCMC, Dict{Int64,Int64}(L))
  beta = [mc.beta[:, i] for i in 1:cbData.I]
  priors = VB.Priors(mc.K, L, use_stickbreak=use_stickbreak)
  c = VB.Constants(cbData.I, cbData.N, cbData.J, mc.K, L,
                   tau, beta, use_stickbreak, noisy_var, priors)
end

println("test state assignment")
Random.seed!(10)
state = VB.State(c)
metrics = Dict{Symbol, Vector{Float64}}()
for m in (:ll, :lp, :lq, :elbo) metrics[m] = Float64[] end
loss(y) = -VB.compute_elbo(state, y, c, metrics) / sum(c.N)
ps = VB.ADVI.vparams(state)
ShowTime() = Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS")

# compute loss
# niters = 10
# loss_y = loss(y)
# println("loss: $(loss_y) | type: $(typeof(loss_y))")
# println("Time $niters elbo computation")
# @time for i in 1:niters
#   idx = [Distributions.sample(1:N[i], minibatch_size, replace=false) for i in 1:I]
#   y_mini = [y[i][idx[i], :] for i in 1:I]
#   loss(y_mini)
# end

#=Test
loss_y = loss(y)
# back!(loss_y)
state.alpha.log_s.tracker.grad
state.alpha.m.tracker.grad
state.H.log_s.tracker.grad
state.H.m.tracker.grad
state.v.log_s.tracker.grad
state.v.m.tracker.grad
state.delta0.log_s.tracker.grad
state.delta0.m.tracker.grad
state.delta1.log_s.tracker.grad
state.delta1.m.tracker.grad
state.eps.log_s.tracker.grad
state.eps.m.tracker.grad
state.W.log_s.tracker.grad
state.W.m.tracker.grad
state.sig2.log_s.tracker.grad
state.sig2.m.tracker.grad
state.y_m.grad
state.y_log_s.grad
=#

# wtf???
println("training...")
opt = ADAM(1e-2)
minibatch_size = 200
niters = 20000
state_hist = typeof(state)[]
for t in 1:niters
  idx = [Distributions.sample(1:c.N[i], minibatch_size, replace=false) for i in 1:c.I]
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


using RCall
println("test rsample of state")
@time realp, tranp, yout, log_qy = VB.rsample(state, y, c);
# samples= [VB.rsample(s, y, c)[2] for s in state_hist[1:4:end]]
samples= [VB.rsample(s, y, c)[2] for s in state_hist]

R"plot"(metrics[:elbo][5:end]/sum(c.N), xlab="iter", ylab="elbo", typ="l")

v = hcat([s.v for s in samples]...).data
v = reshape(v, 1, c.K, length(samples))
H = cat([s.H for s in samples]..., dims=3).data
Z = Int.(v .- H .> 0)

sig2 = hcat([s.sig2 for s in samples]...).data
R"plot"(sig2[1,:], typ="l", xlab="", ylab="")
R"lines"(sig2[2,:])
R"lines"(sig2[3,:])


delta0 = hcat([s.delta0 for s in samples]...).data
R"plot"(delta0[1,:], typ="l", xlab="", ylab="")
