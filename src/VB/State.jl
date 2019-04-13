module dev

using Flux, Flux.Tracker
using Distributions

mutable struct State{A <: AbstractVector}
  delta0::A
  delta1::A
end

mu0(s::State) = -cumsum(s)
mu1(s::State) = cumsum(s)

end # dev

#=
using .dev

x = randn(Float32, 3, 5)
typeof(param(x))


=#
