using Flux, Flux.Tracker
using Distributions

module Dev
include("custom_grads.jl")
include("ModelParam.jl")
include("State.jl")
include("StickBreak.jl")
const SB = StickBreak
end # Dev

# ModelParam.jl
@time s = Dev.ModelParam(Float32, "unit");
@time v = Dev.ModelParam(Float32, 3, "unit");
@time a = Dev.ModelParam(Float32, (3, 5), "unit");

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

delta0 = Dev.ModelParam(Float32, L[0], "positive");
delta1 = Dev.ModelParam(Float32, L[1], "positive");
W = Dev.ModelParam(Float32, (I, K), "simplex");
sig2 = Dev.ModelParam(Float32, I, "positive");
eta0 = Dev.ModelParam(Float32, (I, J, L[0]), "simplex");
eta1 = Dev.ModelParam(Float32, (I, J, L[1]), "simplex");
v = Dev.ModelParam(Float32, K, "unit");
H = Dev.ModelParam(Float32, (J, K), "real");
alpha = Dev.ModelParam(Float32, "real");

state = Dev.State(delta0, delta1, sig2, W, eta0, eta1, v, H, alpha);

samp = Dev.rsample(state)
Dev.SB.transform(samp[:W])
Dev.SB.transform(samp[:eta0])

