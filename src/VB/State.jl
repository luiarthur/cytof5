const TA{F, N} = Tracker.TrackedArray{F, N, Array{F, N}}
const TR{F} = Tracker.TrackedReal{F}

mutable struct State{A1, A2, A3}
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
  y_m::TA{Float64, 2} # I x J
  y_log_s::TA{Float64, 2} # I x J
  
  State(A::Type) = new{A{1}, A{2}, A{3}}()
end

const StateMP = State{ADVI.MPA{Float64, 1}, ADVI.MPA{Float64, 2}, ADVI.MPA{Float64, 3}}

function rsample(s::StateMP, y::Vector{M}, c::Constants; AT::Type=TA{Float64}) where {M}
  real = State(AT)
  tran = State(AT)

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
