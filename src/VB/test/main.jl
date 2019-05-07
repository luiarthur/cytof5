include("gmm.jl")
import Random, Dates

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

Random.seed!(1);
mp = State(K)
loss(y) = -compute_elbo(mp, y, N) / N
ps = ADVI.vparams(mp)

opt = ADAM(1e-1)
minibatch_size = 500
niters = 10000
@time for t in 1:niters
  idx = Distributions.sample(1:N, minibatch_size, replace=false)
  Flux.train!(loss, ps, [(y[idx], )], opt)
  if t % 100 == 0
    println("$(ShowTime()) | $(t)/$(niters) ")
  end
end

samps = [rsample(mp)[2] for i in 1:100]
m_post = vcat([s.m for s in samps]...)
s_post = vcat([s.s for s in samps]...)
w_post = vcat([s.w for s in samps]...)

println("m mean: $(mean(m_post, dims=1)) | true: $m")
println("s mean: $(mean(s_post, dims=1)) | true: $s")
println("w mean: $(mean(w_post, dims=1)) | true: $w")
