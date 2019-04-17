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
