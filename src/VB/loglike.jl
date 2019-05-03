function loglike(s::State{A1, A2, A3}, y::Vector{MA}, c::Constants) where {A1, A2, A3, MA}
  sig = sqrt.(s.sig2)
  noisy_sd = sqrt(c.noisy_var)

  # ll = zero(s.alpha)
  ll = 0
  for i in 1:c.I
    Ni = size(y[i], 1)

    # Ni x J x Lz
    yi = reshape(y[i], Ni, c.J, 1)

    # reshape mu[z] to have size (1 x 1 x L[z])
    mu0 = reshape(-cumsum(s.delta0), 1, 1, c.L[0])
    mu1 = reshape(cumsum(s.delta1), 1, 1, c.L[1])

    # Ni x J x 1
    logmix_L0 = ADVI.lpdf_gmm(yi, mu0, sig[i], s.eta0[i:i, :, :], dims=3, dropdim=false)
    logmix_L1 = ADVI.lpdf_gmm(yi, mu1, sig[i], s.eta1[i:i, :, :], dims=3, dropdim=false)

    # Z: J x K
    # H: J x K
    # v: K
    Z = compute_Z(s.v, s.H, tau=c.tau, use_stickbreak=c.use_stickbreak)
    Z_rs = reshape(Z, 1, c.J, c.K)

    # Ni x J x K -> Ni x K
    Z_mix = ADVI.sumdd(Z_rs .* logmix_L1 .+ (1 .- Z_rs) .* logmix_L0, dims=2)
    # Ni x K
    f = Z_mix .+ log.(s.W[i:i, :])

    # Ni - dimensional
    lli_pre = ADVI.logsumexpdd(f, dims=2)

    # mix with noisy
    lli_quiet = lli_pre .+ log1p(-s.eps[i])
    lli_noisy = ADVI.sumdd(ADVI.lpdf_normal.(y[i], 0, noisy_sd), dims=2) .+ log(s.eps[i])
    # @assert size(lli_quiet) == (size(y[i], 1), )
    # @assert size(lli_noisy) == (size(y[i], 1), )

    # Ni - dimensional
    # TODO: - make stack function
    #       - implement logsumexp(dims=-1)
    lli = ADVI.logsumexpdd(ADVI.stack(lli_quiet, lli_noisy), dims=-1)
    # @assert size(lli) == (size(y[i], 1), )

    # add to ll
    ll += mean(lli) * c.N[i]
  end

  return ll
end
