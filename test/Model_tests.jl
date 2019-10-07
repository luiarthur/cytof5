#=
using Revise, Test
=#
using Cytof5
using Random
using RCall, Distributions
using JLD2, FileIO
using BSON

Random.seed!(10)
printDebug = false

@rimport graphics
@rimport grDevices

function init_state_const_data(; N=[300, 200, 100], J=32, K=4,
                               L=Dict(0=>5, 1=>3))

  I = length(N)
  simdat = Cytof5.Model.genData(J, N, K, L, sortLambda=true)
  d = Cytof5.Model.Data(simdat[:y])
  c = Cytof5.Model.defaultConstants(d, K * 2, Dict(0=>5, 1=>5))
  s = Cytof5.Model.genInitialState(c, d)
  t = Cytof5.Model.Tuners(d.y, c.K)
  X = Float64.(reshape([i for i in 1:I], I, 1))

  return Dict(:d => d, :c => c, :s => s, :t => t, :X => X)
end


@testset "update_W_star" begin
  config = init_state_const_data() 
  cfs = Cytof5.Model.ConstantsFS(config[:c])
  dfs = Cytof5.Model.DataFS(config[:d], config[:X])
  sfs = Cytof5.Model.StateFS{Float64}(config[:s], p=rand(config[:d].I))
  tfs = Cytof5.Model.TunersFS(config[:t], config[:s])


  # Do one update for W_star.
  println("r init: $(sfs.r)")
  println("W_star init: $(sfs.W_star)")
  Cytof5.Model.update_W_star!(sfs, cfs, dfs, tfs)
  println("W_star after: $(sfs.W_star)")
  println("W after: $(sfs.theta.W)")
end


@testset "Compile Model.State." begin
  I = 3
  J = 8
  K = 5
  N = [3, 1, 2] .* 10
  L = Dict{Int, Int}(0 => 5, 1 => 3)

  import Cytof5.Model
  import Cytof5.Model.Cube

  state = Model.State{Float16}()
  state.Z=Matrix{Bool}(undef, J, K)
  state.delta=Dict{Bool, Vector{Float16}}()
  state.alpha=Float16(1.0)
  state.v=ones(Float16, K)
  state.W=rand(Float16, I, K)
  state.sig2=Vector{Float16}(undef, I)
  state.eta=Dict{Bool, Cube{Float16}}()
  state.lam=[ones(Int8, N[i]) for i in 1:I]
  state.gam=[ones(Int8, N[i], J) for i in 1:I]
  state.y_imputed=[randn(Float16, N[i], J) for i in 1:I]
  state.eps=[Float16(.05) for i in 1:I]
  @test true

  # Debug Data constructor
  y = [ randn(N[i], J) for i in 1:I ]
  data = Model.Data(y)
  @test data.I == I
  @test data.J == J
  @test data.N == N

  # Debug Constants constructor
  constants = Model.defaultConstants(data, K, L)
  @test constants.K == K
  @test constants.L == L
end


@testset "Compile Model.genData." begin
  J = 8
  N = [3, 1, 2] * 100
  I = length(N)
  K = 4
  L = Dict{Int, Int}(0 => 5, 1 => 3)
  @time dat = Cytof5.Model.genData(J, N, K, L)

  plot_dat = R"cytof3::plot_dat"
  myImage = R"cytof3::my.image"

  grDevices.pdf("result/simdat_test.pdf")
  myImage(dat[:Z])
  myImage(Cytof5.Model.genZ(J, K, .6))

  for i in 1:I
    for j in 1:J
      plot_dat(dat[:y_complete], i, j)
    end
  end
  grDevices.dev_off()
  @test true
end


@testset "Compile Model.genInitialState." begin
  J = 8
  N = [3, 1, 2] * 100 # Super fast even for 10000 obs. 
  I = length(N)
  K = 4
  # L = Dict{Int, Int}(0 => 5, 1 => 3)
  L = Dict{Int, Int}(0 => 4, 1 => 4)
  @time dat = Cytof5.Model.genData(J, N, K, L, sortLambda=true)
  y_dat = Cytof5.Model.Data(dat[:y])

  R"""
  library(cytof3)
  """

  K_MCMC = K #10
  L_MCMC = L #5
  @time c = Cytof5.Model.defaultConstants(y_dat, K_MCMC, L_MCMC,
                                          noisyDist=Normal(0, sqrt(10)),
                                          yBounds=[-5., -3.5, -2.])
  Cytof5.Model.printConstants(c)
  # Plot miss mech
  R"pdf('result/beta.pdf')"
  y_grid = collect(range(-10, stop=5, length=300))
  R"par(mfrow=c($(y_dat.I), 1))"
  for i in 1:y_dat.I
    p = [Cytof5.Model.prob_miss(yy, c.beta[:, i]) for yy in y_grid]
    R"plot"(y_grid, p, main="i: $i", typ="l",
            xlab="y", ylab="prob of missing", xlim=[-10, 5])
  end
  R"par(mfrow=c(1,1))"
  R"dev.off()"


  # @time init = Cytof5.Model.genInitialState(c, y_dat)
  @time init = Cytof5.Model.smartInit(c, y_dat)
  println("init delta: $(init.delta)")

  printstyled("Test Model Fitting...\n", color=:yellow)
  @time out, lastState, ll, metrics, dden = Cytof5.Model.cytof5_fit(
    init, c, y_dat,
    nmcmc=200, nburn=200,
    computeLPML=true,
    computeDIC=true,
    computedden=true,
    joint_update_Z=false)
  println("Type of dden: $(typeof(dden[end]))")

  println("Type of output: $(typeof(out[1])))")
  Zpost = [o[:Z] for o in out[1]]
  R"pdf('result/Z_post_mean.pdf')"
  R"image"(1 .- mean(Zpost))
  R"dev.off()"

  @save "result/out.jld2" out dat ll lastState
  BSON.bson("result/out.bson", Dict(
    :out => out,
    :dat => dat,
    :ll => ll,
    :lastState => lastState))


  @test true
end
