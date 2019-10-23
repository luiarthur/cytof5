function loglike(s::State{A1, A2, A3}, y::Vector{MA},
                 m::Vector{BitArray{2}}, c::Constants) where {A1, A2, A3, MA}

  # TODO: remove asserts

  sig = sqrt.(s.sig2)
  noisy_sd = sqrt(c.noisy_var)

  # reshape mu[z] to have size (1 x 1 x L[z])
  mu0 = reshape(-cumsum(s.delta0), 1, 1, c.L[0])
  mu1 = reshape(cumsum(s.delta1), 1, 1, c.L[1])

  # Z: J x K
  # H: J x K
  # v: K
  Z = compute_Z(s.v, s.H, tau=c.tau, use_stickbreak=c.use_stickbreak)
  # @assert !isinf(sum(Z))
  # @assert !isnan(sum(Z))
  Z_rs = reshape(Z, 1, c.J, c.K)

  # ll = zero(s.alpha)
  ll = 0.0
  for i in 1:c.I
    mi = m[i]
    Ni = size(y[i], 1)

    # Ni x J x Lz
    yi = reshape(y[i], Ni, c.J, 1)

    # Ni x J x 1
    logmix_L0 = ADVI.lpdf_gmm(yi, mu0, sig[i], s.eta0[i:i, :, :],
                              dims=3, dropdim=false)
    logmix_L1 = ADVI.lpdf_gmm(yi, mu1, sig[i], s.eta1[i:i, :, :],
                              dims=3, dropdim=false)
    # @assert !(isinf(sum(logmix_L1)) || isinf(sum(logmix_L0)))
    # @assert !(isnan(sum(logmix_L1)) || isnan(sum(logmix_L0)))
    @assert size(logmix_L1) == (Ni, c.J, 1) == size(logmix_L1)

    # Ni x J x K -> Ni x K
    Z_mix = ADVI.sumdd(Z_rs .* logmix_L1 + (1 .- Z_rs) .* logmix_L0, dims=2)
    # @assert !isinf(sum(Z_mix))
    # @assert !isnan(sum(Z_mix))
    # Ni x K
    f = Z_mix .+ log.(s.W[i:i, :])
    # @assert !isinf(sum(f))
    # @assert !isnan(sum(f))

    # Ni - dimensional
    lli_pre = ADVI.logsumexpdd(f, dims=2)
    # @assert !isinf(sum(lli_pre))
    # @assert !isnan(sum(lli_pre))

    # mix with noisy
    lli_quiet = lli_pre .+ log1p(-s.eps[i])
    lli_noisy = ADVI.sumdd(ADVI.lpdf_normal.(y[i], 0., noisy_sd), dims=2) .+ log(s.eps[i])
    # @assert size(lli_quiet) == (size(y[i], 1), )
    # @assert size(lli_noisy) == (size(y[i], 1), )

    # Ni - dimensional
    lli = ADVI.logsumexpdd(ADVI.stack(lli_quiet, lli_noisy), dims=-1)
    # @assert !isinf(sum(lli))
    # @assert !isnan(sum(lli))
    # @assert size(lli) == (size(y[i], 1), )

    # p(m | y)
    pm_i = prob_miss(y[i][mi], c.beta[i]...)
    logprob_mi_given_yi = sum(log.(pm_i))

    # add to ll
    fac = c.N[i] / Ni
    ll += (sum(lli) + logprob_mi_given_yi) * fac
  end

  # @assert !isinf(ll)
  # @assert !isnan(ll)

  return ll
end
