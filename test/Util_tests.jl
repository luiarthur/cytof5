#=
using Test
include("../src/Util/Util.jl")
import .Util: similarity_FM, meanabsdiff, sumabsdiff
=#

import Cytof5.Util: similarity_FM, meanabsdiff, sumabsdiff

@testset "Util similarity_FM" begin
  I = 3
  J = 20
  K1 = 5
  K2 = 4

  W1 = rand(I, K1)
  W1 ./= sum(W1, dims=2)

  Z1 = Int.(rand(J, K1) .> .5)
  Z2 = Int.(rand(J, K2) .> .5)

  @time s = similarity_FM(Z1, W1, Z2)
  @assert s <= J * I

  @test true
end
