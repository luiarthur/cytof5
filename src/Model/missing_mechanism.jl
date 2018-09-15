function prob_miss(y, b0, b1)
  return MCMC.sigmoid(b0 - b1 * y)
end
