using Flux, Flux.Tracker
using Distributions
import Dates

include("VB.jl")

# ModelParam.jl
ElType = Float32 # Float64

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
L = Dict(0=>5, 1=>3)
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
state.alpha = VB.ADVI.ModelParam(ElType, (), "positive");

println("test rsample of state")
@time realp, tranp = VB.rsample(state);
sum(tranp.W, dims=2)
sum(tranp.eta0, dims=3)
tranp.delta0
tranp.v
tranp.H
tranp.alpha
tranp.sig2

N = [3, 1, 2] * 2000
I = length(N)
tau = .001
c = VB.Constants{ElType}(I, N, J, K, L, tau, false)

y = [randn(ElType, c.N[i], c.J) for i in 1:c.I]

function elbo(y)
  realp, tranp = VB.rsample(state);
  return VB.loglike(tranp, y, c)
end
loss(y) = -elbo(y) / sum(N)

ps = VB.ADVI.vparams(state)

ShowTime() = Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS")

opt = ADAM(1e-5)
minibatch_size = 500
niters = 10

# compute loss
loss_y = loss(y)
println("loss: $(loss_y) | type: $(typeof(loss_y))")
println("Time $niters elbo computation")
@time for i in 1:niters
  loss(y)
end

#=Test
back!(loss_y)
state.alpha.log_s.tracker.grad
=#

# wtf???
# println("training...")
# for i in 1:niters
#   @time Flux.train!(loss, ps, [(y, )], opt)
#   println("$(ShowTime()) -- $(i)/$(niters)")
# end

