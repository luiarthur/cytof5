using Flux, Flux.Tracker

module Test
using Flux, Flux.Tracker

TA{F, N} = TrackedArray{F, N, Array{F, N}}

mutable struct Bob{F <: AbstractFloat,
                   A1 <: AbstractArray{F, 1} , A2 <: AbstractArray{F, 2}}
  a1::A1
  a2::A2
  Bob(FT::Type, AT::Type) = new{FT, AT{FT, 1}, AT{FT, 2}}()
end

end


# Test
b1 = Test.Bob(Float64, Array)
println(typeof(b1))
b1.a1 = randn(2)
b1.a2 = randn(3, 2)
println(typeof(b1))
 
b1 = Test.Bob(Float64, Test.TA)
println(typeof(b1))
b1.a1 = randn(2)
b1.a2 = randn(3, 2)
println(typeof(b1))


