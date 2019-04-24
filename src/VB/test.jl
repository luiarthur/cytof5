using Flux, Flux.Tracker

module Test
using Flux, Flux.Tracker

TA{F, N} = Tracker.TrackedArray{F, N, Array{F, N}}
TR{F} = Tracker.TrackedReal{F}

mutable struct Bob{F, A1, A2}
  f::F
  a1::A1
  a2::A2
  Bob(F::Type, A::Type) = new{F, A{1}, A{2}}()
end

end


# Test
b1 = Test.Bob(Float64, Array{Float64})
println(typeof(b1))
b1.a1 = randn(2)
b1.a2 = randn(3, 2)
println(typeof(b1))
 
b2 = Test.Bob(Test.TR{Float64}, Test.TA{Float64})
type_pre_alloc = typeof(b2)
b2.f = rand()
b2.a1 = randn(2)
b2.a2 = randn(3, 2)
println(typeof(b2))
@assert type_pre_alloc == typeof(b2)

