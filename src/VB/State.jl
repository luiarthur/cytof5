using Flux, Flux.Tracker
using Distributions

TA{F, N} = Tracker.TrackedArray{F, N, Array{F, N}}
TR{F} = Tracker.TrackedReal{F}

abstract type Advi end
abstract type VP <: Advi end
abstract type RealSpace <: Advi end
abstract type TranSpace <: Advi end

# TODO: upperbound fields with AbstractArray & Real
mutable struct State{T <: Advi, F, A1, A2, A3}
  delta0::A1
  delta1::A1
  sig2::A1
  W::A2
  eta0::A3
  eta1::A3
  v::A1
  H::A2
  alpha::F
  
  State(T::Type, F::Type, A::Type) = new{T, F, A{1}, A{2}, A{3}}()
end

# State{T}() where {T <: Advi} = State{T}(nothing, nothing, nothing,
#                                         nothing, nothing, nothing,
#                                         nothing, nothing, nothing)


function rsample(s::State{VP, MPR{F}, MPA{F, 1}, MPA{F, 2}, MPA{F, 3}};
                 tracking::Bool=true) where {F <: AbstractFloat}
  if tracking
    real = State(RealSpace, TR{F}, TA{F})
    tran = State(TranSpace, TR{F}, TA{F})
  else
    real = State(RealSpace, F, Array{F})
    tran = State(TranSpace, F, Array{F})
  end

  for key in fieldnames(State)
    f = getfield(s, key)
    if typeof(f) <: Array
      # TODO: optimize
      rs = []
      ts = []
      for each_f in f
        println(f)
        r = rsample(f)
        t = transform(f, r)
        append!(rs, r)
        append!(ts, t)
      end
      setfield!(real, key, rs)
      setfield!(tran, key, ts)

    else
      r = rsample(f)
      t = transform(f, r)
      setfield!(real, key, r)
      setfield!(tran, key, t)
    end
  end

  return real, tran
end
