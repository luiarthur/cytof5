using Revise
using Distributions, Random
using Flux, Flux.Tracker
using RCall

@rimport graphics as rgraphics
@rimport rcommon

compute_ke(x) = x^2/2

function hmc(curr_state::Real, log_prob::Function, L::Integer, eps::Real)
  U(q) = -log_prob(q)
  grad_U(q) = Tracker.gradient(U, q)[1]

  q_history = zeros(L + 1)
  p_history = zeros(L + 1)

  q = curr_state
  p = randn()
  
  q_history[1] = q
  p_history[1] = p

  curr_K = compute_ke(p)
  curr_U = U(q)

  p -= eps * Tracker.data(grad_U(q)) / 2
  for i in 1:L
    q += eps * p
    q_history[i + 1] = q
    if i < L
      p -= eps * Tracker.data(grad_U(q))
      p_history[i + 1] = p
    end
  end
  p -= eps * Tracker.data(grad_U(q)) / 2
  p_history[L + 1] = p

  cand_U = U(q)
  cand_K = compute_ke(p)

  log_accept_ratio = curr_U + curr_K - cand_U - cand_K
  
  if log_accept_ratio > log(rand())
    q, lp = q, -cand_U
  else
    q, lp = curr_state, -curr_U
  end

  return Dict(:q => q, :log_prob => lp,
              :qhist => q_history, :phist => p_history)
end


### MAIN ###

function simulate(; L, eps, n, sd=.5, init=0.0, plot_bound=4)
  log_prob(x) = logpdf(Normal(0, sd), x)
  rgraphics.plot(0, xlab="q", ylab="p", typ="n",
                 ylim=[-1, 1]*plot_bound, xlim=[-sd, sd]*plot_bound)
  sim = hmc(init, log_prob, L, eps)
  qs = zeros(n + 1)
  qs[1] = sim[:q]

  for i in 1:n
    sim = hmc(sim[:q], log_prob, L, eps)
    rgraphics.lines(sim[:qhist], sim[:phist], typ="l")
    rgraphics.points(sim[:qhist][end], sim[:phist][end], pch=4, lwd=2)
    qs[i + 1] = sim[:q]
  end

  return qs
end

qs = simulate(L=2^4, eps=.1, n=30, sd=.5)
rcommon.plotPost(qs, typ="l", xlab="", ylab="", main="");


function simulate_gamma(; L, eps, n, sd=.5, init=0.0, plot_bound=4)
  shape = 2
  dist = Gamma(shape, sd/sqrt(shape))
  log_prob(logx) = logpdf(dist, exp(logx)) + logx
  rgraphics.par(mfrow=[2, 1])
  x = collect(range(0, sd*plot_bound, step=.01))
  rgraphics.plot(x, pdf.(dist, x), ylab="", xlab="", main="", typ="l")
  rgraphics.plot(0, xlab="q", ylab="p", typ="n",
                 ylim=[-1, 1]*plot_bound, xlim=[0, sd]*plot_bound)
  sim = hmc(init, log_prob, L, eps)
  qs = zeros(n + 1)
  qs[1] = sim[:q]

  for i in 1:n
    sim = hmc(sim[:q], log_prob, L, eps)
    rgraphics.lines(exp.(sim[:qhist]), sim[:phist], typ="l");
    rgraphics.points(exp(sim[:qhist][end]), sim[:phist][end], pch=4, lwd=2);
    # rgraphics.abline(v=[exp(sim[:qhist][end]), sim[:phist][end]], lty=2, lwd=.5)
    # rgraphics.points(exp(sim[:qhist][1]), sim[:phist][1], pch="$i", lwd=2)
    qs[i + 1] = sim[:q]
  end
  rgraphics.abline(v=mean(dist), lty=2, col="grey")
  rgraphics.par(mfrow=[1, 1])

  return qs
end

Random.seed!(3);
qs = simulate_gamma(L=2^4, eps=.1, n=10, sd=1, plot_bound=5)
rcommon.plotPost(exp.(qs), typ="l", xlab="", ylab="", main="");
_ = nothing

# sim = hmc(0.0, logx -> logpdf(Gamma(2, 3), exp(logx)) + logx, 10, .1)
# rgraphics.plot(sim[:qhist], main="", xlab="", ylab="")
