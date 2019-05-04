using Cytof5
using Flux, Flux.Tracker
using Distributions
import Dates, Random

Random.seed!(0)

include("VB.jl")

# ModelParam.jl
ElType = Float64 # Float64

println("test ModelParam")
@time s = VB.ADVI.ModelParam(ElType, "unit");
@time v = VB.ADVI.ModelParam(ElType, 3, "unit");
@time a = VB.ADVI.ModelParam(ElType, (3, 5), "unit");

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

println("test state assignment")
state = VB.State(VB.ADVI.MPR{ElType}, VB.ADVI.MPA{ElType})
state.delta0 = VB.ADVI.ModelParam(ElType, L[0], "positive");
state.delta1 = VB.ADVI.ModelParam(ElType, L[1], "positive");
state.W = VB.ADVI.ModelParam(ElType, (I, K - 1), "simplex");
state.sig2 = VB.ADVI.ModelParam(ElType, I, "positive");
state.eta0 = VB.ADVI.ModelParam(ElType, (I, J, L[0] - 1), "simplex");
state.eta1 = VB.ADVI.ModelParam(ElType, (I, J, L[1] - 1), "simplex");
state.v = VB.ADVI.ModelParam(ElType, K, "unit");
state.H = VB.ADVI.ModelParam(ElType, (J, K), "unit");
state.alpha = VB.ADVI.ModelParam(ElType, "positive");
state.eps = VB.ADVI.ModelParam(ElType, I, "unit");
state.y_m = param(randn(I, J))
state.y_log_s = param(randn(I, J))

# simulate data
N = [3, 1, 2] * 10000
@time dat = Cytof5.Model.genData(J, N, K, Dict{Int,Int}(L))
I = length(N)
tau = .001
use_stickbreak = false
priors = VB.Priors(K, L, use_stickbreak=use_stickbreak, T=ElType)
noisy_var = 10.0
mc = Cytof5.Model.defaultConstants(Cytof5.Model.Data(dat[:y]), K, Dict{Int64,Int64}(L))
beta = [mc.beta[:, i] for i in 1:I]
c = VB.Constants{ElType}(I, N, J, K, L, tau, beta, use_stickbreak, noisy_var, priors)
y = Matrix{ElType}.(dat[:y])


println("test rsample of state")
@time realp, tranp, yout, log_qy = VB.rsample(state, y, c);
sum(tranp.W, dims=2)
sum(tranp.eta0, dims=3)
tranp.delta0
tranp.v
tranp.H
tranp.alpha
tranp.sig2

loss(y) = -VB.compute_elbo(state, y, c, normalize=true)

ps = VB.ADVI.vparams(state)

ShowTime() = Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS")

opt = ADAM(1e-5)
minibatch_size = 50
niters = 10

# compute loss
loss_y = loss(y)
println("loss: $(loss_y) | type: $(typeof(loss_y))")
println("Time $niters elbo computation")
@time for i in 1:niters
  idx = [Distributions.sample(1:N[i], minibatch_size, replace=false) for i in 1:I]
  y_mini = [y[i][idx[i], :] for i in 1:I]
  loss(y_mini)
end

#=Test
back!(loss_y)
state.alpha.log_s.tracker.grad
=#

# wtf???
println("training...")
for i in 1:niters
  idx = [Distributions.sample(1:N[i], minibatch_size, replace=false) for i in 1:I]
  y_mini = [y[i][idx[i], :] for i in 1:I]
  @time Flux.train!(loss, ps, [(y_mini, )], opt)
  println("$(ShowTime()) -- $(i)/$(niters)")
end

