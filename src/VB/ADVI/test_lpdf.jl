using Distributions

include("ADVI.jl")

# Dirichlet
d = Dirichlet(collect(1:3)); x = collect(1:3) / sum(1:3)
@assert ADVI.compute_lpdf(d, x) == logpdf(d, x)

# Normal
d = Normal(2, 3); x = -5.0
@assert ADVI.compute_lpdf(d, x) == logpdf(d, x)

# Beta
d = Beta(2, 3); x = .6
@assert ADVI.compute_lpdf(d, x) == logpdf(d, x)

# Gamma
d = Gamma(2,4); x = 5.0
@assert ADVI.compute_lpdf(d, x) == logpdf(d, x)

# LogNormal
d = LogNormal(2, 4); x = 1.5
@assert ADVI.compute_lpdf(d, x) == logpdf(d, x)
# ADVI.compute_lpdf(d, x)
# logpdf(d, x)

# Uniform
d = Uniform(2,4); x = 3.5
@assert ADVI.compute_lpdf(d, x) == logpdf(d, x)

