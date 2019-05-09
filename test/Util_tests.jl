import Cytof5.Util: similarity_FM, meanabsdiff, sumabsdiff

@testset "Util similarity_FM" begin
  I = 3
  J = 20
  K1 = 40
  K2 = 50

  W1 = rand(I, K1)
  W2 = rand(I, K2)

  Z1 = Int.(rand(J, K1) .> .5)
  Z2 = Int.(rand(J, K2) .> .5)

  @time similarity_FM(Z1, Z2, W1, W2, z_diff=meanabsdiff)

  @test true
end
