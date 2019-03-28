using Flux, Flux.Tracker
using Distributions

struct VP
  b0
  b1

  VP() = new(param(randn(2)), param(randn(2)))
end

sigmoid(x) = 1.0 / (1.0 + exp(-x))

function loglike(y, x, params, N)
  b0 = params[:b0]
  b1 = params[:b1]
  prob = clamp.(sigmoid.(b0 .+ b1 .* x), .001, .999)
  # return sum(y .* log.(prob) .+ (1 .- y) .* log1p.(-prob)) * (N / length(y))
  return sum(logpdf.(Bernoulli.(prob), y)) * (N / length(y))
end

function log_p(params)
  # return 0.0
  b0 = params[:b0]
  b1 = params[:b1]
  return logpdf(Normal(0, 10), b0) + logpdf(Normal(0, 10), b1)
end

lpdf_vp(p, x) = logpdf(Normal(p[1], exp(p[2])), x)
function log_q(params, vp)
  return lpdf_vp(vp.b0, params[:b0]) + lpdf_vp(vp.b1, params[:b1])
end

rsample(p) = randn() * exp(p[2]) + p[1]
rsample(vp::VP) = Dict(:b0 => rsample(vp.b0), :b1 => rsample(vp.b1))

struct Metrics
  ll::Float64
  lp::Float64
  lq::Float64
end

function elbo(y, x, vp, N, metrics::Vector{Metrics})
  params = rsample(vp)

  ll = loglike(y, x, params, N)
  lp = log_p(params)
  lq = log_q(params, vp)

  m = Metrics(ll.data, lp.data, lq.data)
  push!(metrics, m)

  return ll + lp - lq
end
