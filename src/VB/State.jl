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
  alpha::A1 # 1 (F won't work, TrackedReals don't work as expected)
  eps::A1 # I
  y_m::TA{Float64, 2} # I x J
  y_log_s::TA{Float64, 2} # I x J
  
  State(A::Type) = new{A{1}, A{2}, A{3}}()
end

const StateMP = State{ADVI.MPA{Float64, 1}, ADVI.MPA{Float64, 2}, ADVI.MPA{Float64, 3}}

function State(c::Constants)
  s = State(ADVI.MPA{Float64})
  s.delta0 = ADVI.ModelParam(c.L[0], "positive")
  s.delta1 = ADVI.ModelParam(c.L[1], "positive")
  s.W = ADVI.ModelParam((c.I, c.K - 1), "simplex")
  s.sig2 = ADVI.ModelParam(c.I, "positive", m=param(fill(-1.0, c.I)), log_s=param(fill(-1.0, c.I)))
  s.eta0 = ADVI.ModelParam((c.I, c.J, c.L[0] - 1), "simplex")
  s.eta1 = ADVI.ModelParam((c.I, c.J, c.L[1] - 1), "simplex")
  s.v = ADVI.ModelParam(c.K, "unit")
  s.H = ADVI.ModelParam((c.J, c.K), "unit")
  s.alpha = VB.ADVI.ModelParam("positive")
  s.eps = ADVI.ModelParam(c.I, "unit", m=param(fill(-3., c.I)), log_s=param(fill(-3., c.I)))
  s.y_m = param(fill(-3.0, c.I, c.J))
  s.y_log_s = param(fill(log(.1), c.I, c.J))

  return s
end


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
  yout = AT[]
  log_qy = 0
  vae = VAE(s.y_m, s.y_log_s)
  for i in 1:c.I
    yi, log_qyi = vae(i, y[i])
    append!(yout, [yi])
    log_qy += log_qyi * c.N[i] / size(y[i], 1)
  end

  return real, tran, yout, log_qy
end
