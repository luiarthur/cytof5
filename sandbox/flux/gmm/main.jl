include("model.jl")
using Flux, Flux.Tracker
import Random

Random.seed!(0);

# Generate Data
N = 1000
m = [1., -3., 5.]
s = [.2, .4, .1]
w = [.3, .45, .25]
K = length(m)
y = zeros(N)
for i in 1:N
  k = rand(Categorical(w))
  y[i] = rand(Normal(m[k], s[k]))
end

vp = VP(K)
function loss(y::T) where T
  return -elbo(y, vp) / N
end
params = Tracker.Params([getfield(vp, fn) for fn in fieldnames(typeof(vp))])

opt = ADAM(1e-1)
minibatch_size = 100
niters = 10000

# Flux.train!(loss, params, [(y, )], opt)

Random.seed!(3);
@time for i in 1:niters
  idx = sample(1:N, minibatch_size)
  Flux.train!(loss, params, [(y[idx], )], opt)
  if i % 10 == 0
    println("Progress: $(i)/$(niters) | ELBO: $(-loss(y).data)")
    println(vp.m[:, 1])
    println(exp.(vp.log_s[:, 1]))
    println(softmax_fullrank(vp.real_w[:, 1]))
  end
end

s_post = vcat([exp.(rsample(vp.log_s.data)) for b in 1:100]'...)
w_post = vcat([softmax_fullrank(rsample(vp.real_w)) for b in 1:100]'...)
println("m_mean: $(vp.m[:, 1].data) | s_mean: $(mean(s_post, dims=1)), | w_mean: $(mean(w_post, dims=1))")
