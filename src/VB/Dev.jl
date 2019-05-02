using Flux, Flux.Tracker
using Distributions
import Dates

include("VB.jl")

# ModelParam.jl
ElType = Float32 # Float64

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

delta0 = VB.ADVI.ModelParam(ElType, L[0], "positive");
delta1 = VB.ADVI.ModelParam(ElType, L[1], "positive");
W = VB.ADVI.ModelParam(ElType, (I, K - 1), "simplex");
sig2 = VB.ADVI.ModelParam(ElType, I, "positive");
eta0 = VB.ADVI.ModelParam(ElType, (I, J, L[0] - 1), "simplex");
eta1 = VB.ADVI.ModelParam(ElType, (I, J, L[1] - 1), "simplex");
v = VB.ADVI.ModelParam(ElType, K, "unit");
H = VB.ADVI.ModelParam(ElType, (J, K), "unit");
alpha = VB.ADVI.ModelParam(ElType, 1, "positive");

# state = VB.State{VB.VP, VB.ModelParam, VB.ModelParam, VB.ModelParam, VB.ModelParam}(delta0, delta1, sig2, W, eta0, eta1, v, H, alpha);
state = VB.State(VB.VP,
                 VB.ADVI.MPR{ElType},
                 VB.ADVI.MPA{ElType})
state.delta0=delta0
state.delta1=delta1
state.W=W
state.sig2=sig2
state.eta0=eta0
state.eta1=eta1
state.v=v
state.H=H
state.alpha=alpha

@time realp, tranp = VB.rsample(state);
sum(tranp.W, dims=2)
sum(tranp.eta0, dims=3)
tranp.delta0
tranp.v
tranp.H
tranp.alpha
tranp.sig2

N = [3, 1, 2] * 100
I = length(N)
tau = .1
c = VB.Constants(I, N, J, K, L, tau, false)

y = [randn(c.N[i], c.J) for i in 1:c.I]

function elbo(y)
  realp, tranp = VB.rsample(state);
  return VB.loglike(tranp, y, c)
end
loss(y) = -elbo(y) / sum(N)

ps = []
for fn in fieldnames(typeof(state))
  f = getfield(state, fn)
  append!(ps, [f.m])
  append!(ps, [f.log_s])
end
ps = Tracker.Params(ps)

ShowTime() = Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS")

opt = ADAM(1e-5)
minibatch_size = 500
niters = 10

# wtf???
# println("training...")
# for i in 1:niters
#   @time Flux.train!(elbo, ps, [(y, )], opt)
#   println("$(ShowTime()) -- $(i)/$(niters)")
# end
