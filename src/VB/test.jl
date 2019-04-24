using Flux, Flux.Tracker

struct Theta
  delta0
  delta1
  sig2
  W
  eta0
  eta1
  v
  H
  alpha
end

(d::Dict)(x) = println(x)

##########################
using Flux, Flux.Tracker
module Test

mutable struct Bob{T <: AbstractFloat, A <: AbstractArray{T}}
  a::A
  Bob(TT::Type, AA::Type) = new{TT, AA{TT}}()
end
end

b1 = Test.Bob(Float16, Array)
b1.a = randn(2)
println(typeof(b1))

b2 = Test.Bob(Float32, TrackedArray)
b2.a = param(randn(Float32, 3))
println(typeof(b2))
