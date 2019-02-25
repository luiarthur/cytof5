using Flux, Flux.Tracker
using Distributions

struct VP
  b0
  b1

  VP() = new(param(randn(2)), param(randn(2)))
end

sigmoid(x) = 1.0 / (1.0 + exp(-x))

function loglike(y, x, b0, b1, N)
  prob = clamp.(sigmoid.(b0 .+ b1 .* x), .001, .999)
  return sum(y .* log.(prob) .+ (1 .- y) .* log1p.(-prob)) * (N / length(y))
  # return sum(logpdf.(Bernoulli.(prob), y)) * (N / length(y))
end

function log_p(b0, b1)
  # return logpdf(Normal(0, 1), b0) + logpdf(Normal(0, 1), b1)
  return 0.0
end

lpdf_vp(p, x) = logpdf(Normal(p[1], exp(p[2])), x)
function log_q(b0, b1, vp)
  return lpdf_vp(vp.b0, b0) + lpdf_vp(vp.b1, b1)
end

rsample(p) = randn() * exp(p[2]) + p[1]
function elbo(y, x, vp, N)
  b0 = rsample(vp.b0)
  b1 = rsample(vp.b1)

  return loglike(y, x, b0, b1, N) + log_p(b0, b1) - log_q(b0, b1, vp)
end

