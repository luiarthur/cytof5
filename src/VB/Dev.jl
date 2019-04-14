using Flux, Flux.Tracker
using Distributions

module Dev
include("custom_grads.jl")
include("ModelParam.jl")
include("State.jl")
end # Dev

# ModelParam.jl
ElType = Float32 # Float64

@time s = Dev.ModelParam(ElType, "unit");
@time v = Dev.ModelParam(ElType, 3, "unit");
@time a = Dev.ModelParam(ElType, (3, 5), "unit");

@time Dev.vp(s);
@time Dev.vp(v);
@time Dev.vp(a);

@time Dev.rsample(s);
@time Dev.rsample(v);
@time Dev.rsample(a);

# State
L = Dict(0=>5, 1=>3)
I = 3
J = 20
K = 4

delta0 = Dev.ModelParam(ElType, L[0], "positive");
delta1 = Dev.ModelParam(ElType, L[1], "positive");
W = Dev.ModelParam(ElType, (I, K), "simplex");
sig2 = Dev.ModelParam(ElType, I, "positive");
eta0 = Dev.ModelParam(ElType, (I, J, L[0]), "simplex");
eta1 = Dev.ModelParam(ElType, (I, J, L[1]), "simplex");
v = Dev.ModelParam(ElType, K, "unit");
H = Dev.ModelParam(ElType, (J, K), "real");
alpha = Dev.ModelParam(ElType, "real");

state = Dev.State(delta0, delta1, sig2, W, eta0, eta1, v, H, alpha);

samp = Dev.rsample(state)
sum(samp[:W].tran, dims=2)
sum(samp[:eta0].tran, dims=3)
samp[:delta0]
samp[:v].tran
