function loglike(s::State{TranSpace, F, A1, A2, A3},
                 y::Vector{Matrix{Float64}}, c::Constants) where {F, A1, A2, A3}
  sig = sqrt.(s.sig2)

  # ll = zero(s.alpha)
  ll = 0
  for i in 1:c.I
    Ni = size(y[i], 1)

    # Ni x J x Lz
    yi = reshape(y[i], Ni, c.J, 1)

    # reshape mu[z] to have size (1 x 1 x L[z])
    mu0 = reshape(-cumsum(s.delta0), 1, 1, c.L[0])
    mu1 = reshape(cumsum(s.delta1), 1, 1, c.L[1])

    # Ni x J
    logmix_L0 = ADVI.lpdf_gmm(yi, mu0, sig[i], s.eta0[i:i, :, :], dims=3)
    logmix_L1 = ADVI.lpdf_gmm(yi, mu1, sig[i], s.eta1[i:i, :, :], dims=3)

    # Z: J x K
    # H: J x K
    # v: K
    # c: Ni x J x K
    # d: Ni x K
    # Ni x J x K

    v = c.use_stickbreak ? cumprod(s.v, dims=1) : s.v
    Z = compute_Z(reshape(v, 1, c.K), s.H, c.tau)
    Z_rs = reshape(Z, 1, c.J, c.K)
    lg0_rs = reshape(logmix_L0, Ni, c.J, 1)
    lg1_rs = reshape(logmix_L1, Ni, c.J, 1)

    # Ni x 1
    Z_mix = sum(Z_rs .* lg1_rs .+ (1 .- Z_rs) .* lg0_rs, dims=2)
    f = Z_mix .+ log.(s.W[i:i, :])

    # Ni - dimensional
    lli = ADVI.logsumexp(f, dims=2)

    # add to ll
    ll += mean(lli) * c.N[i]
  end

  return ll
end
