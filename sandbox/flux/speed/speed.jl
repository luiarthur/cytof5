using Flux, Flux.Tracker
using Distributions
import Random

function loglike_fast(y, m, log_s)
  s = exp(log_s[1])
  return sum(logpdf.(Normal(m[1], s), y))
end

function loglike_slow(y, m, log_s)
  n = length(y)
  s = exp(log_s[1])
  out = 0.0
  dist = Normal(m[1], s)
  for i in 1:n
    out += logpdf(dist, y[i])
  end
  return out
end

### Main ###
Random.seed!(0);

# Generate Data
m_true = 2.3
s_true = 0.5
# N = 100000 # This casues a stack overflow in the slow version! This is due to
# recursive nature of the backprop. See
# https://github.com/FluxML/Flux.jl/issues/509
N = 10000
y = rand(Normal(m_true, s_true), N)


function experiment(loglike; niter=500, lr=1e-1, seed=0, show_output=true)
  Random.seed!(0);
  m = param(randn(1))
  log_s = param(randn(1))
  ps = Flux.params(m, log_s) 
  loss(y) = -loglike(y, m, log_s)
  opt = ADAM(lr)

  @time for i in 1:niter
    Flux.train!(loss, ps, [(y, )], opt)
  end

  if show_output
    println("m: $(m[1].data) | s: $(exp(log_s[1]).data)")
  end
end

# Compile
println("Compile functions")
experiment(loglike_fast, niter=1, show_output=false);
experiment(loglike_slow, niter=1, show_output=false);
println()

# FAST
println("Run fast version")
experiment(loglike_fast, niter=50)
println()

# SLOW
println("Run slow version")
experiment(loglike_slow, niter=50)
println()

# CONCLUSION: Use vectorized code when possible. This is easier for the computational
#             graphs created for backprop. Smaller graph -> faster & less memory usage.
