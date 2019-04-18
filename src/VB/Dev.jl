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

state = VB.State{VB.VP}(delta0, delta1, sig2, W, eta0, eta1, v, H, alpha);

realp, tranp = VB.rsample(state)
sum(tranp.W, dims=2)
sum(tranp.eta0, dims=3)
tranp.delta0
tranp.v
tranp.H
tranp.alpha
tranp.sig2

# Model Constants
struct Constants
  I
  N
  J
  K
  L
  tau
  use_stickbreak
end

N = [3, 1, 2] * 10
I = length(N)
tau = .1
c = Constants(I, N, J, K, L, tau, false)

y = [randn(c.N[i], c.J) for i in 1:c.I]
ll = VB.loglike(tranp, y, c)
@time back!(ll)
W.m.grad
