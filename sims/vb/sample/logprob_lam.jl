include("util.jl")
import Cytof5.VB.ADVI

function logprob_lam(s::StateArray, y::Vector{Matrix{Float64}}, c::Cytof5.VB.Constants)
  # Get params
  eps = s.eps
  W = s.W
  sig = sqrt.(s.sig2)
  eta0 = s.eta0
  eta1 = s.eta1

  # reshape mu[z] to have size (1 x 1 x L[z])
  mu0 = reshape(-cumsum(s.delta0), 1, 1, c.L[0])
  mu1 = reshape(cumsum(s.delta1), 1, 1, c.L[1])

  # Compute Z
  H = s.H
  v = s.v
  if c.use_stickbreak
    Z = Int.(reshape(cumprod(v), 1, c.K) .> H)
  else
    Z = Int.(reshape(v, 1, c.K) .> H)
  end

  noisy_sd = sqrt(c.noisy_var)

  # Compute loglike
  function engine(i::Int)
    Ni = size(y[i], 1)

    # Ni x J x Lz
    yi = reshape(y[i], Ni, c.J, 1)

    # Ni x J x 1
    logmix_L0 = ADVI.lpdf_gmm(yi, mu0, sig[i], eta0[i:i, :, :], dims=3, dropdim=false)
    logmix_L1 = ADVI.lpdf_gmm(yi, mu1, sig[i], eta1[i:i, :, :], dims=3, dropdim=false)

    # Ni x J x K -> Ni x K
    Z_rs = reshape(Z, 1, c.J, c.K)
    Z_mix = ADVI.sumdd(Z_rs .* logmix_L1 + (1 .- Z_rs) .* logmix_L0, dims=2)

    # Ni x K
    lpi_quite = (Z_mix .+ log.(W[i:i, :])) .+ log1p(-s.eps[i])

    # Ni x 1
    lpi_noisy = ADVI.sumdd(ADVI.lpdf_normal.(y[i], 0., noisy_sd), dims=2) .+ log(s.eps[i])

    return [lpi_quite lpi_noisy]
  end

  return [engine(i) for i in 1:c.I]
end
