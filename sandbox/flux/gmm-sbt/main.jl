include("model.jl")
using Flux, Flux.Tracker
import Random
import Dates

Random.seed!(1);
ShowTime() = Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS")

# Generate Data
N = 1000
m = [3., -1., 2.]
s = [.1, .05, .15]
w = [.5, .3, .2]
K = length(m)
y = zeros(N)
for i in 1:N
  k = rand(Categorical(w))
  y[i] = rand(Normal(m[k], s[k]))
end

K = 3
vp = VP(K)
loss_hist = zeros(0)
function loss(y::T) where T
  out = -elbo(y, vp) / N
  append!(loss_hist, out.data)
  return out
end
params = Tracker.Params([getfield(vp, fn) for fn in fieldnames(typeof(vp))])

opt = ADAM(1e-1)
minibatch_size = 100
niters = 1300


Random.seed!(0);
@time for i in 1:niters
  idx = sample(1:N, minibatch_size)
  Flux.train!(loss, params, [(y[idx], )], opt)
  if i % 100 == 0

    println("$(ShowTime()) -- $(i)/$(niters) -- ELBO: $(-loss_hist[end])")
    # println(vp.m[:, 1])
    # println(exp.(vp.log_s[:, 1]))
  end
end

s_post = vcat([exp.(rsample(vp.log_s.data)) for b in 1:100]'...)
w_post = vcat([StickBreak.transform(rsample(vp.real_w)) for b in 1:100]'...)

println("m_mean: $(vp.m[:, 1].data) | m: $(m)")
println("s_mean: $(mean(s_post, dims=1)) | s: $(s)")
println("w_mean: $(mean(w_post, dims=1)) | w: $(w)")
