function betaBinomial(prior::P, bernoulliTrials::Vector{Int}) where {P <: Beta}
  n = length(bernoulliTrials)
  successes = sum(bernoulliTrials)
  failures = n - successes
  a = prior.α + successes
  b = prior.β + failures
  return Beta(a, b)
end
#betaBinomial(Beta(2,3), [1,0,1,1])

