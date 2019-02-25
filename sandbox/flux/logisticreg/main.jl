include("model.jl")
using Flux, Flux.Tracker
import Random

Random.seed!(0);

# Generate Data
N = 1000
x = randn(N)
b0 = .5
b1 = 2.0
p = sigmoid.(b0 .+ b1 * x)
y = (p .> rand(N)) * 1.0

minibatch_size = 100
vp = VP()
loss(y, x) = -elbo(y, x, vp, N) / N

params = Tracker.Params([getfield(vp, fn) for fn in fieldnames(typeof(vp))])
# grads = Tracker.gradient(() -> loss(y, x), params)

opt = ADAM(1e-1)
niters = 1000

@time for i in 1:niters
  idx = sample(1:N, minibatch_size)
  Flux.train!(loss, params, [(y[idx], x[idx])], opt)
  if i % (niters / 10) == 0
    println("Progress: $(i)/$(niters) | ELBO: $(-loss(y, x).data)")
  end
end

println("b0_mean: $(vp.b0[1].data) | b1_mean: $(vp.b1[1].data)")
println("b0_sd: $(exp(vp.b0[2].data)) | b1_sd: $(exp(vp.b1[2].data))")
