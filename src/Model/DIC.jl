# Functions for computing DIC
# http://webpages.math.luc.edu/~ebalderama/myfiles/modelchecking101_pres.pdf

# TODO: Test

function getSuffStats(s::State)
  I = length(s.y_imputed)
  J = size(s.Z, 1)
  N = length.(s.lam)

  p = [ [prob_miss(init.y_imputed[i][n, j], init.b0[i], init,b1[i]) for n in 1:N[i], j in 1:J] for i in 1:I]
  mus = [ [s.mus[s.Z[j, s.lam[i][n]]][s.gam[i][n, j]] for n in 1:N[i], j in 1:J] for i in 1:I]
  sig = sqrt.(s.sig2)

  return (p, mus, sig)
end

mutable struct DicSuffStats
  p::Vector{Matrix{Float64}}
  mus::Vector{Matrix{Float64}}
  sig::Vector{Float64}

  Dsum::Float64
  counter::Int

  function DicSuffStats(init::State)
    p, mus, sig = getSuffStats(init)
    new(p, mus, sig, 0, 0)
  end
end

function updateDicSuffStats(d::DicSuffStats, s::State)
  p, mus, sig = getSuffStats(s)

  if d.counter > 0
    d.p += p
    d.mus += mus
    d.sig += sig
    d.Dsum += deviance(p, mus, sig)
  else
    @assert d.counter == 0
    d.p = p
    d.mus = mus
    d.sig = sig
    d.Dsum = deviance(p, mus, sig)
  end

  d.counter += 1
end

function deviance(p::Vector{Matrix{Float64}},
                  mus::Vector{Matrix{Float64}},
                  sig::Vector{Float64}, dat::Data)
  ll = 0

  for i in 1:dat.I
    for j in 1:dat.J
      for n in 1:dat.N[i]
        ll += pdf(Bernoulli(p[i][n, j]), dat.m[i][n, j])
        if dat.m[i][n, j] == 0
          ll += pdf(Normal(mus[i][n, j]), sig[i][n, j], dat.y[i][n, j])
        end
      end
    end
  end

  return -2 * ll
end

function computeDIC(d::DicSuffStats, dat::Data, return_Dmean_and_pD)
  pMean = d.p ./ d.counter
  musMean = d.mus ./ d.counter
  sigMean = d.sig ./ d.counter
  Dmean = d.Dsum / d.counter
  pD = Dmean - deviance(pMean, musMean, sigMean, dat)

  if return_Dmean_and_pD
    return Dmean, pD
  else
    return Dmean + pD
  end
end
