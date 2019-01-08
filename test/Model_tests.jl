Random.seed!(10)
printDebug = false
using RCall, Distributions
using JLD2, FileIO

@testset "Compile Model.State." begin
  I = 3
  J = 8
  K = 5
  N = [3, 1, 2] .* 10
  L = Dict{Int, Int}(0 => 5, 1 => 3)

  import Cytof5.Model
  import Cytof5.Model.Cube

  Model.State(Z=Matrix{Bool}(undef, J, K),
              mus=Dict{Bool, Vector{Float16}}(),
              alpha=Float16(1.0),
              v=ones(Float16, K),
              W=rand(Float16, I, K),
              sig2=Vector{Float16}(undef, I),
              eta=Dict{Bool, Cube{Float16}}(),
              lam=[ones(Int8, N[i]) for i in 1:I],
              gam=[ones(Int8, N[i], J) for i in 1:I],
              y_imputed=[randn(Float16, N[i], J) for i in 1:I],
              eps=[Float16(.05) for i in 1:I])
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

  R"pdf('result/simdat_test.pdf')"
  myImage(dat[:Z])
  myImage(Cytof5.Model.genZ(J, K, .6))

  for i in 1:I
    for j in 1:J
      plot_dat(dat[:y_complete], i, j)
    end
  end
  R"dev.off()"
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
  @time c = Cytof5.Model.defaultConstants(y_dat, K_MCMC, L_MCMC)
  Cytof5.Model.printConstants(c)
  # Plot miss mech
  R"pdf('result/beta.pdf')"
  y_grid = collect(range(-10, stop=5, length=300))
  R"par(mfrow=c($(y_dat.I), 1))"
  for i in 1:y_dat.I
    p = [Cytof5.Model.prob_miss(yy, c.beta[:, i]) for yy in y_grid]
    R"plot"(y_grid, p, main="i: $i", typ="l", xlab="y", ylab="prob of missing", xlim=[-10, 5])
  end
  R"par(mfrow=c(1,1))"
  R"dev.off()"


  # @time init = Cytof5.Model.genInitialState(c, y_dat)
  @time init = Cytof5.Model.smartInit(c, y_dat)

  printstyled("Test Model Fitting...\n", color=:yellow)
  @time out, lastState, ll = Cytof5.Model.cytof5_fit(init, c, y_dat,
                                                     nmcmc=200, nburn=200,
                                                     computeLPML=true,
                                                     computeDIC=true)

  println("typeof output: $(typeof(out[1])))")
  Zpost = [o[:Z] for o in out[1]]
  R"pdf('result/Z_post_mean.pdf')"
  R"image"(1 .- mean(Zpost))
  R"dev.off()"

  @save "result/out.jld2" out dat ll lastState

  #=
  using JLD2, FileIO, RCall

  R"""
  my.image($(dat[:y][1]), col=blueToRed(11), zlim=c(-3,3), addL=T, na.col='black')
  """
  @load "result/out.jld2" out dat ll lastState
  R"plot"(ll, type="l", ylab="")
  Zpost = [o[:Z] for o in out[1]]
  Zmean = zeros(size(Zpost[1]));
  for z in Zpost
    Zmean .+= z / length(Zpost)
  end

  myImage = R"cytof3::my.image"

  for i in 1:length(Zpost)
    z = Zpost[i]
    sleep(.1)
    myImage(z, main=i)
  end

  R"pdf('result/cytof_test.pdf')"
  myImage(Zmean)
  myImage(dat[:Z])

  lastState.sig2
  lastState.eta
  dat[:eta]
  lastState.W - dat[:W]
  lastState.lam[1]
  dat[:lam][1]

  plotPosts = R"rcommon::plotPosts"

  
  R"my.image($(lastState.y_imputed[1]), col=blueToRed(11), addL=T, zlim=c(-5,5))"
  R"dev.off()"
  =#


  @test true
end
