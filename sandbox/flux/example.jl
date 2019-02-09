# https://fluxml.ai/Flux.jl/stable/training/optimisers/
using Flux, Flux.Tracker
using Distributions

struct VarParam
  m::Flux.Tracker.TrackedReal{Float64}
  log_s::Flux.Tracker.TrackedReal{Float64}
  VarParam() = VarParam(param(0.0), param(0.0))
end

get_vd(vp) = Normal(vp.m, exp(vp.log_s))

function logpdf_logx(logx::Float64, logpdf)
  x = exp(logx)
  return logpdf(x) + logx
end

function rsample(vp::VarParam)
  return randn() * exp(vp.log_s) + vp.m
end

function loglike(y, x, params)
  b0 = params[:b0]
  b1 = params[:b1]
  sig = params[:sig]

  return sum(logpdf.(Normal.(b0 .+ b1 .* x, sig), y))
end

lpdf_sig(x) = logpdf(Gamma(1, 1), x)

function log_prior(real_params)
  b0 = real_params[:b0]
  b1 = real_params[:b1]
  log_sig = real_params[:sig]

  return logpdf(Normal(0, 1), b0) + logpdf(Normal(0, 1), b1) + logpdf_logx(log_sig, lpdf_sig)
end

function log_q(real_params, vp)
  b0 = params[:b0]
  b1 = params[:b1]
  sig = params[:sig]
 
  return logpdf(get_vd(vp[:b0]), b0) + logpdf(get_vd(vp[:b1]), b1) +
         logpdf(get_vd(vp[:sig]), log(sig))
end

function to_param_space(real_params)
  params = deepcopy(real_params)
  params[:sig] = exp(params[:sig])
  return params
end

vp = VarParam(param(0.0), param(0.0))
rsample(vp)
lpdf(vp, 2.0)

N = 1000
x = randn(N)
b0 = 2.0
b1 = -3.0
sig = 0.5
y = b0 .+ b1 .* x .+ sig


vps = Dict(:b0 => VarParam(), :b1 => VarParam(), :sig => VarParam())
# grads = Tracker.gradient(() -> loss(x, y), θ)

opt = ADAM(.1)
