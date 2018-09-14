Random.seed!(10)
printDebug = false
using RCall

@testset "Compile Model.State." begin
  I = 3
  J = 8
  K = 5
  N = [3, 1, 2] .* 10
  L = 4

  import Cytof5.Model
  import Cytof5.Model.Cube

  Z=Matrix{Int8}(undef, J, K)
  mus=Dict{Tuple{Int8,Int8}, Matrix{Float16}}()
  alpha=1.0
  v=fill(1.0, K)
  W=rand(I,K)
  sig2=Vector{Float16}(undef, I)
  eta=Dict{Int8, Cube{Float16}}()
  lam=[fill(1, N[i]) for i in 1:I]
  gam=[ones(Int, N[i], J) for i in 1:I]
  y_imputed=[randn(N[i], J) for i in 1:I]
  b0=1.0
  b1=1.0

  Model.State(Z, mus, alpha, v, W, sig2, eta, lam, gam, y_imputed, b0, b1)
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
  I = 3
  J = 8
  N = [3, 1, 2] * 100
  K = 4
  L = 5
  @time dat = Cytof5.Model.genData(I, J, N, K, L)

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
  I = 3
  J = 8
  N = [3, 1, 2] * 10000 # Super fast even for 10000 obs. 
  K = 4
  L = 5
  @time dat = Cytof5.Model.genData(I, J, N, K, L)
  y_dat = Cytof5.Model.Data(dat[:y])
  @time c = Cytof5.Model.defaultConstants(y_dat, K, L)
  @time init = Cytof5.Model.genInitialState(c, y_dat)
  @test true
end
