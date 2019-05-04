const TA{F, N} = Tracker.TrackedArray{F, N, Array{F, N}}
const TR{F} = Tracker.TrackedReal{F}

mutable struct State{F, A1, A2, A3}
  delta0::A1 # L0
  delta1::A1 # L1
  sig2::A1 # I
  W::A2 # I x K
  eta0::A3 # I x J x K
  eta1::A3 # I x J x K
  v::A1 # K
  H::A2 # J x K
  # alpha::F # 1 (F won't work, TrackedReals don't work as expected)
  alpha::A1 # 1 (F won't work, TrackedReals don't work as expected)
  eps::A1 # I
  y_m # I x J
  y_log_s # I x J
  
  State(F::Type, A::Type) = new{F, A{1}, A{2}, A{3}}()
end

const StateMP{F} = State{ADVI.MPR{F}, ADVI.MPA{F, 1}, ADVI.MPA{F, 2}, ADVI.MPA{F, 3}} where {F <: AbstractFloat} 

(s::StateMP{F})(AT::Type=TA{F}) where {F <: AbstractFloat} = rsample(s, AT=AT)

function rsample(s::StateMP{F}, y::Vector{M}, c; AT::Type=TA{F}) where {F <: AbstractFloat, M}
  FT = typeof(s.alpha.m)
  real = State(FT, AT)
  tran = State(FT, AT)

  for key in fieldnames(State)
    if !(key in (:y_m, :y_log_s))
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
  end

  # Draw y and compute log q(y|m) 
  yout = []
  log_qy = 0
  vae = VAE(s.y_m, s.y_log_s)
  for i in 1:c.I
    yi, log_qyi = vae(i, y[i])
    append!(yout, [yi])
    log_qy += log_qyi * c.N[i] / size(y[i], 1)
  end

  return real, tran, yout, log_qy
end
