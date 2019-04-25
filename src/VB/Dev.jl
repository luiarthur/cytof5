using Flux, Flux.Tracker
using Distributions

# module Dev
# include("custom_grads.jl")
# include("ModelParam.jl")
# include("State.jl")


include("VB.jl")
# end # Dev

# ModelParam.jl
ElType = Float32 # Float64

@time s = VB.ModelParam(ElType, "unit");
@time v = VB.ModelParam(ElType, 3, "unit");
@time a = VB.ModelParam(ElType, (3, 5), "unit");

@time VB.vp(s);
@time VB.vp(v);
@time VB.vp(a);

@time VB.rsample(s);
@time VB.rsample(v);
@time VB.rsample(a);

# State
L = Dict(0=>5, 1=>3)
I = 3
J = 20
K = 4

delta0 = VB.ModelParam(ElType, L[0], "positive");
delta1 = VB.ModelParam(ElType, L[1], "positive");
W = VB.ModelParam(ElType, (I, K - 1), "simplex");
sig2 = VB.ModelParam(ElType, I, "positive");
eta0 = VB.ModelParam(ElType, (I, J, L[0] - 1), "simplex");
eta1 = VB.ModelParam(ElType, (I, J, L[1] - 1), "simplex");
v = VB.ModelParam(ElType, K, "unit");
H = VB.ModelParam(ElType, (J, K), "real");
alpha = VB.ModelParam(ElType, "real");

# state = VB.State{VB.VP, VB.ModelParam, VB.ModelParam, VB.ModelParam, VB.ModelParam}(delta0, delta1, sig2, W, eta0, eta1, v, H, alpha);
state = VB.State(VB.VP,
                 VB.MPR{ElType},
                 VB.MPA{ElType})
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

N = [3, 1, 2] * 10000
I = length(N)
tau = .1
c = VB.Constants(I, N, J, K, L, tau, false)

y = [randn(c.N[i], c.J) for i in 1:c.I]
ll = VB.loglike(tranp, y, c)
println("Time ll computation...")
for q in 1:3
  @time ll = VB.loglike(tranp, y, c)
end
@time back!(ll)
W.m.grad;
