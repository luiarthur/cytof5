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


function loglike(params, c, data, idx)
  println("NotImplemented")
  y = params[:y]
  sig = sqrt.(params[:sig2])

  ll = zero(params[:alpha])
  for i in 1:c.I
    # Ni x J x Lz
    yi = reshape(y[i], c.N[i], c.J, 1)

    mu0 = reshape(-cumsum(params[:delta0], 1, 1, c.L[0]))
    lf0 = logpdf.(Normal(mu0, sig[i]), yi) .+ log.(params[:eta0][i:i, :, :])

    mu1 = reshape(-cumsum(params[:delta1], 1, 1, c.L[1]))
    lf1 = logpdf.(Normal(mu1, sig[i]), yi) .+ log.(params[:eta1][i:i, :, :])

    # Ni x J
    logmix_L0 = logsumexp(lf0, 3)
    logmix_L1 = logsumexp(lf1, 3)

    # Z: J x K
    # H: J x K
    # v: K
    # c: Ni x J x K
    # d: Ni x K
    # Ni x J x K

    if c.use_stickbreak
      v = cumprod(params[:v])
    else
      v = params[:v]
    end


  end

  return ll
end


function logprior(reals, params)
  println("NotImplemented")
end


function logq(reals)
  println("NotImplemented")
end

