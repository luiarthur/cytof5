include("gmm.jl")
import Random, Dates
using RCall
Random.seed!(0);

ShowTime() = Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS")

# Generate Data
N = 10000
m = [3., -1., 2.]
s = [.1, .05, .15]
w = [.5, .3, .2]
K = length(m)
y = [begin
       k = rand(Categorical(w))
       rand(Normal(m[k], s[k]))
     end for i in 1:N]

metrics = Dict(m => Float64[] for m in (:elbo, :ll, :lp, :lq))

# TRAIN_VERSION = 1
TRAIN_VERSION = 2
if TRAIN_VERSION == 1
  Random.seed!(6); # VERSION I
else
  Random.seed!(0); # VERSION II
  # Random.seed!(100); # VERSION II
end
mp = State(K)
loss(y) = -compute_elbo(mp, y, N, metrics) / N
ps = ADVI.vparams(mp)

opt = ADAM(1e-1)
# opt = RMSProp(1e-1)
minibatch_size = 500
niters = 1000

if TRAIN_VERSION == 1
  # VERSION I
  @time Flux.train!(loss, ps, [(y[sample(1:N, minibatch_size, replace=false)], )
                               for t in 1:niters], opt)
else
  @time for t in 1:niters
    idx = Distributions.sample(1:N, minibatch_size, replace=false)

    # Flux.train!(loss, ps, [(y[idx], )], opt)
    # OR
    gs = Tracker.gradient(() -> loss(y[idx]), ps)
    Flux.Tracker.update!(opt, ps, gs)
    #= looking at each grad
    for k in keys(gs.grads)
      println(gs.grads[k])
    end
    =#

    if t % 100 == 0
      println("$(ShowTime()) | $(t)/$(niters) ")
    end
  end
end

println(metrics[:elbo][1:100:end] / N)
R"plot"(metrics[:elbo]/N, typ="l", xlab="", ylab="")

samps = [rsample(mp)[2] for i in 1:100]
m_post = cat([s.m for s in samps]..., dims=2)
s_post = cat([s.s for s in samps]..., dims=2)
w_post = vcat([s.w for s in samps]...)

println("m mean: $(mean(m_post, dims=2)) | true: $m")
println("s mean: $(mean(s_post, dims=2)) | true: $s")
println("w mean: $(mean(w_post, dims=1)) | true: $w")
