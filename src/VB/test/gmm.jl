using Flux, Flux.Tracker
using Distributions

include("../ADVI/ADVI.jl")

mutable struct State
  w
  m
  s
  State() = new()
end

function State(K::Integer)
  state = State()

  # NOTE: this works! Notice that dimensions do matter
  #       especially for dirichlet stuff. Needs to be a tuple for size.
  #       So, the minimum must be (1, K-1) as the last dim is the dirichlet
  #       dim. Maybe needs a FIXME?
  state.m = ADVI.ModelParam(K, "real")
  state.w = ADVI.ModelParam((1, K - 1), "simplex",
                            m=fill(0.0, 1, K-1),
                            s=fill(exp(-2), 1, K-1))
  state.s = ADVI.ModelParam(K, "positive",
                            m=fill(0.0, K),
                            s=fill(exp(-2), K))

  return state
end

function rsample(state::State)
  real = State()
  tran = State()

  for key in fieldnames(State)
    f = getfield(state, key)

    r = ADVI.rsample(f)
    t = ADVI.transform(f, r)

    setfield!(real, key, r)
    setfield!(tran, key, t)
  end

  return real, tran
end

function loglike(tran::State, y::Vector{Float64})
  K = length(tran.m)
  N = length(y)

  # NOTE: Dirichlet parameters need special treatment because it changes dims!
  w = reshape(tran.w, 1, K)
  m = reshape(tran.m, 1, K)
  s = reshape(tran.s, 1, K)
  y_rs = reshape(y, N, 1)
  return sum(ADVI.lpdf_gmm(y_rs, m, s, w, dims=2, dropdim=true))
end

function logprior(real::State, tran::State, mp::State)
  lp = 0.0

  lp += sum(ADVI.compute_lpdf(Normal(1.85, 5), tran.m))

  lp += sum(ADVI.compute_lpdf(Gamma(1, 0.1), tran.s))
  lp += sum(ADVI.logabsdetJ(mp.s, real.s, tran.s))

  K = length(tran.w)
  lp += sum(ADVI.compute_lpdf(Dirichlet(ones(K) / K), tran.w))
  lp += sum(ADVI.logabsdetJ(mp.w, real.w, tran.w))

  return lp
end

function logq(real::State, mp::State)
  lq = 0.0
  for key in fieldnames(State)
    vp = getfield(mp, key)
    r = getfield(real, key)
    lq += sum(ADVI.log_q(vp, r))
  end
  return lq
end

function compute_elbo(mp::State, y::Vector{Float64}, N::Integer,
                      metrics::Dict{Symbol, Vector{Float64}})
  real, tran = rsample(mp)

  ll = loglike(tran, y) * N / length(y)
  lp = logprior(real, tran, mp)
  lq = logq(real, mp)

  elbo = ll + lp - lq
  append!(metrics[:elbo], elbo.data)

  return elbo
end
