using Flux, Flux.Tracker
using Distributions

struct State
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

function rsample(s::State)
  out = Dict{Symbol, Any}()

  for key in fieldnames(State)
    f = getfield(s, key)
    if typeof(f) <: Array
      out[key] = [rsample(each_f) for each_f in f]
    else
      out[key] = rsample(f)
    end
  end

  return out
end
