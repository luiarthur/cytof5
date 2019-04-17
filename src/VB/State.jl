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


struct Sample
  real
  tran
end


function rsample(s::State)
  out = Dict{Symbol, Sample}()

  for key in fieldnames(State)
    f = getfield(s, key)
    if typeof(f) <: Array
      out[key] = [begin
                    real = rsample(each_f)
                    tran = transform(f, real)
                    Sample(real, tran)
                  end for each_f in f]
    else
      real = rsample(f)
      tran = transform(f, real)
      out[key] = Sample(real, tran)
    end
  end

  return out
end



