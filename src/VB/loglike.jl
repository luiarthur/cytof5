function lpdf_normal(x::X, m::M, s::S) where {X <: Real, M <: Real, S<:Real}
  z = (x - m) / s
  return -0.5 * log(2*pi) - z^2 * 0.5 - log(s)
end

function loglike(s::State{TranSpace}, y, c)
  sig = sqrt.(s.sig2)

  ll = zero(s.alpha)
  for i in 1:c.I
    Ni = size(y[i], 1)

    # Ni x J x Lz
    yi = reshape(y[i], Ni, c.J, 1)

    mu0 = reshape(-cumsum(s.delta0), 1, 1, c.L[0])
    # lf0 = logpdf.(Normal.(mu0, sig[i]), yi) .+ log.(s.eta0[i:i, :, :])
    lf0 = lpdf_normal.(yi, mu0, sig[i]) .+ log.(s.eta0[i:i, :, :])

    mu1 = reshape(cumsum(s.delta1), 1, 1, c.L[1])
    # lf1 = logpdf.(Normal.(mu1, sig[i]), yi) .+ log.(s.eta1[i:i, :, :])
    lf1 = lpdf_normal.(yi, mu1, sig[i]) .+ log.(s.eta1[i:i, :, :])

    # Ni x J
    logmix_L0 = SB.logsumexp(lf0, dims=3)
    logmix_L1 = SB.logsumexp(lf1, dims=3)

    # Z: J x K
    # H: J x K
    # v: K
    # c: Ni x J x K
    # d: Ni x K
    # Ni x J x K

    v = c.use_stickbreak ? cumprod(s.v, dims=1) : s.v
    Z = compute_Z(reshape(v, 1, c.K) .- cdf.(Normal(0, 1), s.H), c.tau)
    Z_rs = reshape(Z, 1, c.J, c.K)
    lg0_rs = reshape(logmix_L0, Ni, c.J, 1)
    lg1_rs = reshape(logmix_L1, Ni, c.J, 1)

    # Ni x 1
    Z_mix = sum(Z_rs .* lg1_rs .+ (1 .- Z_rs) .* lg0_rs, dims=2)
    f = Z_mix .+ log.(s.W[i:i, :])

    # Ni - dimensional
    lli = SB.logsumexp(f, dims=2)

    # add to ll
    ll += mean(lli) * c.N[i]
  end

  return ll
end
