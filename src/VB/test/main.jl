include("gmm.jl")
import Random, Dates

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

mp = State(K)
loss(y) = -compute_elbo(mp, y, N) / N
ps = ADVI.vparams(mp)

opt = ADAM(1e-2)
minibatch_size = 500
niters = 10000
for t in 1:niters
  idx = Distributions.sample(1:N, minibatch_size, replace=false)
  Flux.train!(loss, ps, [(y[idx], )], opt)
  # if t % 10 == 0
    println("$(ShowTime()) | $(t)/$(niters) ")
  # end
end


