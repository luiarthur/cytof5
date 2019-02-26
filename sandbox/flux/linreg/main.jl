include("model.jl")
using Flux, Flux.Tracker
import Random

Random.seed!(0);

# Generate Data
N = 1000
x = randn(N)
b0 = 2.0
b1 = -3.0
sig = 0.5
y = b0 .+ b1 .* x .+ randn(N) * sig

minibatch_size = 100
vp = VP()
loss(y, x) = -elbo(y, x, vp, N) / N

params = Tracker.Params([getfield(vp, fn) for fn in fieldnames(typeof(vp))])
# grads = Tracker.gradient(() -> loss(y, x), params)

opt = ADAM(1e-1)
niters = 10000

@time for i in 1:niters
  idx = sample(1:N, minibatch_size)
  Flux.train!(loss, params, [(y[idx], x[idx])], opt)
  if i % (niters / 10) == 0
    println("Progress: $(i)/$(niters) | ELBO: $(-loss(y, x).data)")
  end
end

sig_post = exp.([rsample(vp.log_sig) for b in 1:1000])
println("b0_mean: $(vp.b0[1].data) | b1_mean: $(vp.b1[1].data) | sig_mean: $(mean(sig_post).data)")
println("Loss: $(loss(y, x).data)")
