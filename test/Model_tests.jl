Random.seed!(10)
printDebug = false

@testset "Compile Model.State." begin
  I = 3
  J = 8
  K = 5
  N = [3, 1, 2] .* 10

  import Cytof5.Model
  import Cytof5.Model.Cube

  Model.State(Z=Matrix{Int8}(undef, J, K),
              mus=Dict{Int8, Matrix{Float16}}(),
              alpha=1.0,
              v=fill(1.0, K),
              W=rand(I,K),
              sig2=Vector{Float16}(undef, I),
              eta=Dict{Int8, Cube{Float16}}(),
              lam=[fill(1, N[i]) for i in 1:I],
              gam=[randn(N[i], J) for i in 1:I],
              y_imputed=[randn(N[i], J) for i in 1:I],
              b0=1.0,
              b1=1.0);
  @test true

  # Debug Data constructor
  y = [ randn(N[i], J) for i in 1:I ]
  Model.Data(y)
  @test true

  # Debug Constants constructor
end
