println("Loading Packges for Cytof5 test...")
using Cytof5
using Test
using LinearAlgebra
using Distributions
using RCall
println("Starting Tests for Cytof5 test...")

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
  function update(s::State)
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

@testset "MCMC.metropolisAdaptie" begin
  muTrue = [1., 3.]
  sig2True = .5
  nHalf = 300
  y = [ rand(Normal(m, sqrt(sig2True)), nHalf) for m in muTrue ]
  n = nHalf * 2

  muPriorMean = 0.
  muPriorSd = 5.

  # TODO
end
