function genInitialState(c::Constants, d::Data)
  J = d.J
  K = c.K
  L = c.L
  I = d.I
  N = d.N

  vec_y = vcat(vec.(d.y)...)
  y_neg = filter(y_inj -> !isnan(y_inj) && y_inj < 0, vec_y)

  y_imputed = begin
    local out = [zeros(Float64, N[i], J) for i in 1:I]
    # y_lower, y_upper = quantile.(c.mus_prior[0], [0, .1])
    y_lower, y_upper = quantile(y_neg, [0, .1])
    for i in 1:I
      for n in 1:N[i]
        for j in 1:J
          if isnan(d.y[i][n, j])
            # out[i][n, j] = rand(c.mus_prior[0])
            out[i][n, j] = rand(Uniform(y_lower, y_upper))
          else
            out[i][n, j] = d.y[i][n, j]
          end
          @assert !isnan(out[i][n, j])
        end
      end
    end

    out
  end

  alpha = rand(c.alpha_prior)
  v = rand(Beta(alpha / c.K, 1), K)
  Z = [ Bool(rand(Bernoulli(v[k]))) for j in 1:J, k in 1:K ]
  mus = Dict([Bool(z) => sort(rand(c.mus_prior[z], L[z])) for z in 0:1])
  sig2 = [rand(c.sig2_prior) for i in 1:I]
  W = Matrix{Float64}(hcat([ rand(c.W_prior) for i in 1:I ]...)')
  lam = [ Int8.(rand(Categorical(W[i,:]), N[i])) for i in 1:I ]
  eta = begin
    function gen(z)
      arrMatTo3dArr([ rand(c.eta_prior[z]) for i in 1:I, j in 1:J ])
    end
    Dict([Bool(z) => gen(z) for z in 0:1])
  end
  gam = [zeros(Int8, N[i], J) for i in 1:I]
  for i in 1:I
    for j in 1:J
      for n in 1:N[i]
        z_lin = Z[j, lam[i][n]]
        gam[i][n, j] = rand(Categorical(eta[z_lin][i, j, :]))
      end
    end
  end

  return State(Z=Z, mus=mus, alpha=alpha, v=v, W=W, sig2=sig2, 
               eta=eta, lam=lam, gam=gam, y_imputed=y_imputed)
end
