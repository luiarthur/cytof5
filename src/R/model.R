model.code = function() nimbleCode({
  for (k in 1:K) {
    v[k] ~ dbeta(alpha/K, 1)
    for (j in 1:J) {
      Z0[j, k] ~ dbern(v[k])
      Z[j, k] <- Z0[j, k] + 1
    }
  }

  alpha ~ dgamma(a_alpha, b_alpha)

  for (l in 1:L) {
    # mus_0
    mus[1,l] ~ T(dnorm(psi_0, var=tau2_0), -30, 0)
    # mus_1
    mus[2,l] ~ T(dnorm(psi_1, var=tau2_1), 0, 30)
  }

  for (i in 1:I) {
    W[i,1:K] ~ ddirch(d_W[1:K])
    sig2[i] ~ dinvgamma(a_sig, b_sig)
    for (j in 1:J) {
      eta[1, i, j, 1:L] ~ ddirch(a_eta0[1:L]) # eta_0
      eta[2, i, j, 1:L] ~ ddirch(a_eta1[1:L]) # eta_1
    }
  }

  # Likelihood
  for (n in 1:N_sum) {
    lam[n] ~ dcat(W[get_i[n], 1:K])  
    for (j in 1:J) {
      gam[n, j] ~ dcat(eta[Z[j, lam[n]], get_i[n], j, 1:L])
      y[n, j] ~ dnorm(mus[Z[j, lam[n]], gam[n, j]], var=sig2[get_i[n]])
      m[n, j] ~ dbern(p[n, j])
      logit(p[n, j]) <- b0 + b1 * y[n, j]
    }
  }

  b0 ~ dnorm(m_b0, var=s2_b0)
  b1 ~ dnorm(m_b1, var=s2_b1)
            
  # Constrains mu to be in ascending order
  # TODO: These make sampling a little harder
  #       Can I remove them at the expense of less identifiability?
  constraints_mus0 ~ dconstraint( prod(mus[1, 1:(L-1)] <= mus[1, 2:L]) )
  constraints_mus1 ~ dconstraint( prod(mus[2, 1:(L-1)] <= mus[2, 2:L]) )
})

