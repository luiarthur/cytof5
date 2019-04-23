using Flux, Flux.Tracker
using Distributions

TS(T) = typeof(param(rand(T)))
TV(T) = typeof(param(rand(T, 0)))
TM(T) = typeof(param(rand(T, 0, 0)))
TC(T) = typeof(param(rand(T, 0, 0, 0)))

abstract type Advi end
abstract type VP <: Advi end
abstract type RealSpace <: Advi end
abstract type TranSpace <: Advi end

mutable struct State{T <: Advi}
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

State{T}() where {T <: Advi} = State{T}(nothing, nothing, nothing,
                                        nothing, nothing, nothing,
                                        nothing, nothing, nothing)


function rsample(s::State{VP})
  real = State{RealSpace}()
  tran = State{TranSpace}()

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
