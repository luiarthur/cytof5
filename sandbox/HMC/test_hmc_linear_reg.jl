#=
using Revise
=#

include("HMC.jl")
using Distributions, Random
using Flux

using RCall
@rimport graphics as rgraphics
@rimport rcommon

struct State
  beta
  log_sig2
end

pretty(s::State) = "State(β => $(s.beta); σ => $(sqrt(exp(s.log_sig2[1]))))"

State(K::Int) = State(param(zeros(K)), param(ones(1)))

function sim_data(;N::Int=100, K::Int=10, sig=.3)
  X = randn(N, K)
  b = randn(K)
  y = X * b + randn(N) * sig
  return Dict(:X => X, :b => b, :y => y, :sig => sig)
end


function lpdf_std_normal(z)
  return -z ^ 2 / 2 - log(2 * pi) / 2
end

lpdf_normal(y, m, s) = lpdf_std_normal((y-m)/s) - log(s)

function loglike(X, y, state::State)
  # println(state)
  m = X * state.beta
  s = sum(sqrt.(exp.(state.log_sig2))) + 1e-6
  # s = 1.0
  ll = sum(lpdf_normal.(y, m, s))
  # println("ll: $(ll)")
  return ll
end

function logprior(state::State)
  lp_beta = sum(logpdf.(Normal(0, 10), state.beta))
  lp_sig2 = sum(logpdf.(InverseGamma(2, 1), exp.(state.log_sig2)))
  lp_sig2 += sum(state.log_sig2)  # add log abs jacobian
  lp = lp_beta + lp_sig2
  # println("lp: $(lp)")
  return lp
end

### MAIN ###
Random.seed!(1)

N, K = (1000, 20)
dat = sim_data(K=K)
log_prob(state::State) = loglike(dat[:X], dat[:y], state) + logprior(state)
state = State(K)
epsilon = 5 / N
num_leapfrog_steps = 100

# Compile
_ = HMC.hmc_update(state, log_prob, num_leapfrog_steps, epsilon)

function simulate(init; nburn=500, nsamps=1000)
  state = deepcopy(init)

  log_prob_hist = zeros(nsamps)
  samps = [state for i in 1:nsamps]
  for i in 1:(nburn + nsamps)
    print("\rProgress: $i / $(nburn + nsamps)")
    state, log_prob_curr = HMC.hmc_update(state, log_prob,
                                          num_leapfrog_steps, epsilon)
    if i > nburn
      samps[i - nburn] = state
      log_prob_hist[i - nburn] = log_prob_curr
    end
  end
  return samps, log_prob_hist
end

samps, log_prob_hist = simulate(state, nburn=500, nsamps=500)
println(pretty(samps[end]));
dat

acceptance_rate = length(unique(log_prob_hist)) / length(log_prob_hist)
println("Acceptance rate: $acceptance_rate")
rgraphics.plot(log_prob_hist, main="", xlab="iter", ylab="log prob", typ="l");

sig2_post = [exp(Tracker.data(s.log_sig2[1])) for s in samps]
b_first_2_post = cat([Tracker.data(s.beta[1:2])' for s in samps]..., dims=1)
rgraphics.plot(b_first_2_post, xlab="", ylab="", main="", typ="l");

rcommon.plotPosts(b_first_2_post);
rcommon.plotPost(sqrt.(sig2_post));
rgraphics.abline(v=dat[:sig], lwd=3, col="orange");
