TA{F, N} = Tracker.TrackedArray{F, N, Array{F, N}}
TR{F} = Tracker.TrackedReal{F}

mutable struct State{A1, A2, A3}
  delta0::A1 # L0
  delta1::A1 # L1
  sig2::A1 # I
  W::A2 # I x K
  eta0::A3 # I x J x K
  eta1::A3 # I x J x K
  v::A1 # K
  H::A2 # J x K
  alpha::A1 # 1 (F won't work, TrackedReals don't work as expected)
  # y_ms_fn::A2 # I x J
  
  State(A::Type) = new{A{1}, A{2}, A{3}}()
end

function rsample(s::State{ADVI.MPA{F, 1}, ADVI.MPA{F, 2}, ADVI.MPA{F, 3}};
                 AT::Type=TA{F}) where {F <: AbstractFloat}

  real = State(AT)
  tran = State(AT)

  for key in fieldnames(State)
    f = getfield(s, key)
    if typeof(f) <: Array
      # TODO: optimize
      rs = []
      ts = []
      for each_f in f
        println(f)
        r = ADVI.rsample(f)
        t = ADVI.transform(f, r)
        append!(rs, r)
        append!(ts, t)
      end
      setfield!(real, key, rs)
      setfield!(tran, key, ts)

    else
      r = ADVI.rsample(f)
      t = ADVI.transform(f, r)
      setfield!(real, key, r)
      setfield!(tran, key, t)
    end
  end

  return real, tran
end
