Random.seed!(10)
printDebug = false

@testset "Compile MCMC.metropolis." begin
  lfc(x) = x
  MCMC.metropolis(1.0, lfc, 1.0)
  @test true

  lfc(x::Vector{Float64}) = sum(x)
  MCMC.metropolis(randn(4), lfc, Matrix{Float64}(I,4,4))
  @test true
end

@testset "Compile MCMC.TuningParam." begin
  tval_init = 1.23
  t = MCMC.TuningParam(tval_init)
  MCMC.update(t, true)
  MCMC.update(t, false)
  MCMC.update(t, true)
  @test t.value == tval_init
  @test MCMC.acceptanceRate(t) == 2/3
end

mutable struct State
  x::Int
  y::Float64
  z::Vector{Float64}
end

@testset "Compile MCMC.gibbs." begin
  function update(s::State, i::Int, out::Any)
    s.x += 1
    s.y -= 1
    s.z[1] += 1
  end

  s = State(0, 0, [0,0])
  out, lastState = MCMC.gibbs(s, update, monitors=[[:x,:y], [:z]], thins=[1,2])
  @test true
end

@testset "Test logit / sigmoid" begin
  @test MCMC.logit(.6) ≈ log(.6 / .4)
  @test MCMC.sigmoid(3.) ≈ 1 / (1 + exp(-3.))
end

mutable struct Param
  mu::Vector{Float64}
  sig2::Float64
end

@testset "MCMC.metropolisAdaptive" begin
  muTrue = [1., 3.]
  J = length(muTrue)
  sig2True = .5
  nHalf = 300
  y = [ rand(Normal(m, sqrt(sig2True)), nHalf) for m in muTrue ]
  n = nHalf * 2

  muPriorMean = 0.
  muPriorSd = 5.

  sig2Prior_a = 3.
  sig2Prior_b = 2.

  tunerMu = [ deepcopy(MCMC.TuningParam(1.0)) for j in 1:J ]
  tunerSig2 = MCMC.TuningParam(1.0)

  function update(s::Param, i::Int, out::Any)

    function update_muj(j::Int)
      function lfc(muj::Float64)
        ll = sum(logpdf.(Normal(muj, sqrt(s.sig2)), y[j]))
        lp = logpdf(Normal(muPriorMean, sqrt(muPriorSd)), muj)
        return ll + lp
      end

      s.mu[j] = MCMC.metropolisAdaptive(s.mu[j], lfc, tunerMu[j])
      return
    end

    function update_mu()
      for j in 1:J
        update_muj(j)
      end
      return
    end

    function update_sig2()
      function lfc(log_sig2::Float64) 
        sig2 = exp(log_sig2)
        ll = 0
        for j in 1:J
          ll += sum(logpdf.(Normal(s.mu[j], sqrt(sig2)), y[j]))
        end
        lp = MCMC.logpdfLogInverseGamma(log_sig2, sig2Prior_a, sig2Prior_b)
        return ll + lp
      end

      s.sig2 = exp(MCMC.metropolisAdaptive(log(s.sig2), lfc, tunerSig2))
      return
    end

    # Update
    update_mu()
    update_sig2()

    return
  end

  init = Param([0., 0.], sig2True)
  @time out, state = MCMC.gibbs(deepcopy(init), update, nmcmc=2000, nburn=10000)

  if printDebug
    # Print acceptance rates
    for j in 1:J
      println("μ$j Acceptance rate: $(MCMC.acceptanceRate(tunerMu[j]))")
      println("μ$j Tuner value:     $(tunerMu[j].value)")
    end
    println("sig2 Acceptance rate:  $(MCMC.acceptanceRate(tunerSig2))")
    println("sig2 Tuner value:      $(tunerSig2.value)")

    # Print Means
    muPost = hcat([o[:mu] for o in out[1]]...)
    sig2Post = hcat([o[:sig2] for o in out[1]]...)
    R"""
    library(rcommon)
    pdf('result/amcmc.pdf')
    plotPosts($(muPost'))
    plotPost($(sig2Post'))
    dev.off()
    """
  end
end

