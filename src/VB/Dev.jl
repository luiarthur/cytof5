using Cytof5
using Flux, Flux.Tracker
using Distributions
import Dates, Random

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

# State
L = Dict(false=>5, true=>3)
I = 3
J = 20
K = 4


println("Simulate Data")
N = [3, 1, 2] * 10000
@time dat = Cytof5.Model.genData(J, N, K, Dict{Int,Int}(L))
I = length(N)
tau = .01 # FIXME: why can't I do .001?
use_stickbreak = false
priors = VB.Priors(K, L, use_stickbreak=use_stickbreak)
noisy_var = 10.0
mc = Cytof5.Model.defaultConstants(Cytof5.Model.Data(dat[:y]), K, Dict{Int64,Int64}(L))
beta = [mc.beta[:, i] for i in 1:I]
c = VB.Constants(I, N, J, K, L, tau, beta, use_stickbreak, noisy_var, priors)
y = dat[:y]

println("test state assignment")
Random.seed!(10)
state = VB.State(c)
elbo_hist = Float64[]
loss(y) = -VB.compute_elbo(state, y, c, elbo_hist, normalize=true)
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
back!(loss_y)
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
niters = 200
state_hist = typeof(state)[]
for t in 1:niters
  idx = [Distributions.sample(1:N[i], minibatch_size, replace=false) for i in 1:I]
  y_mini = [y[i][idx[i], :] for i in 1:I]
  Flux.train!(loss, ps, [(y_mini, )], opt)
  if t % 10 == 0
    println("$(ShowTime()) -- $(t)/$(niters)")
    append!(state_hist, [deepcopy(state)])
  end
end


using RCall
println("test rsample of state")
@time realp, tranp, yout, log_qy = VB.rsample(state, y, c);
samples= [VB.rsample(s, y, c)[2] for s in state_hist[1:4:end]]

R"plot"(elbo_hist[5:end], xlab="iter", ylab="elbo", typ="l")

sig2 = hcat([s.sig2 for s in samples]...).data
R"plot"(sig2[1,:], typ="l", xlab="", ylab="")
R"lines"(sig2[2,:])
R"lines"(sig2[3,:])


delta0 = hcat([s.delta0 for s in samples]...).data
R"plot"(delta0[1,:], typ="l", xlab="", ylab="")
