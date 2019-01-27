using Random
using Distributions
using Knet

Random.seed!(0);

struct LM
  b0
  b1
  sig
end



function f(x, b0, b1)
  return b0 .+ b1 * x
end

function loglike(b0, b1, sig, x, y)
  # return sum(logpdf.(Normal.(f(x, b0, b1), sig), y))
  return -log(sig) - sum(abs2, y - f(x, b0, b1)) / (2 * sig ^2)
end

# DATA
N = 100
x = randn(N)
b0_true = 2.0
b1_true = 3.0
sig_true = .4
y = f(x, b0_true, b1_true) .+ randn(N) * sig_true

# SGD training loop:

function mini(x, y, n)
  idx = rand(1:N, n)
  return (x[idx], y[idx])
end

grad_ll = grad(loglike)

b0 = 0.0
b1 = 0.0
log_sig = 0.0
lr = 1e-2
for i in 1:1000
  g = grad(loglike)(b0, b1, exp(log_sig), x, y)
end
