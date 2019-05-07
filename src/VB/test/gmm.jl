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
  #       especially for dirichlet stuff
  state.w = ADVI.ModelParam((1, K - 1), "simplex")
  state.m = ADVI.ModelParam(K, "real")
  state.s = ADVI.ModelParam(K, "positive")

  # NOTE: this works, but I don't like it
  # state.w = ADVI.ModelParam((1, K - 1), "simplex")
  # state.m = ADVI.ModelParam((1, K), "real")
  # state.s = ADVI.ModelParam((1, K), "positive")

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

  # NOTE: This does not work!
  m = reshape(tran.m, 1, K)
  s = reshape(tran.s, 1, K)
  w = reshape(tran.w, 1, K)
  y_rs = reshape(y, N, 1)
  return sum(ADVI.lpdf_gmm(y_rs, m, s, w, dims=2, dropdim=true))

  # NOTE: this works, but I don't like it
  # return sum(ADVI.lpdf_gmm(reshape(y, N, 1), tran.m, tran.s, tran.w,
  #                          dims=2, dropdim=true))
end

function logprior(real::State, tran::State, mp::State)
  lp = 0.0

  lp += sum(ADVI.compute_lpdf(Normal(0, 10), tran.m))
  # lp += sum(ADVI.lpdf_normal.(tran.m, 0.0, 10.0))

  lp += sum(ADVI.compute_lpdf(Gamma(1, 1), tran.s))
  # lp += sum(ADVI.lpdf_gamma.(tran.s, 1.0, 1.0))
  lp += sum(ADVI.logabsdetJ(mp.s, real.s, tran.s))

  K = length(tran.w)
  lp += sum(ADVI.compute_lpdf(Dirichlet(ones(K)), tran.w))
  # lp += sum(ADVI.lpdf_dirichlet(tran.w, one.(tran.w)))
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

function compute_elbo(mp::State, y::Vector{Float64}, N::Integer)
  real, tran = rsample(mp)

  ll = loglike(tran, y) * N / length(y)
  lp = logprior(real, tran, mp)
  lq = logq(real, mp)

  elbo = ll + lp - lq
  # println("elbo: $(elbo / N)")

  return elbo
end
