"""
ConstantsFS object for FAM with feature selection.
Augments Constants object with new priors:
W_star_prior, p_prior, and omega_prior.
Note that only one of p_prior or omega_prior will be used. 
"""
@namedargs mutable struct ConstantsFS
  W_star_prior::Gamma  # W*_ik ~ Gamma(a_W, 1)
  p_prior::Beta  # p_c ~ Beta(a_p, b_p)
  omega_prior::ContinuousDistribution  # omage_each ~ Normal(m_omega , s_omega)
  constants::Constants
  
  function ConstantsFS(c::Constants)
    ws_prior = Gamma(1. / c.K, 1)
    p_prior = Beta(.1, .1)
    omega_prior = Normal(0., 10)
    return new(ws_prior, p_prior, omega_prior)
  end
end
