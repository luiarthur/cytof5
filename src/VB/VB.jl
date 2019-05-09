module VB

using Distributions
using Flux, Flux.Tracker

import Random # shuffle, seed
import Dates
ShowTime() = Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS")

import Cytof5.Model: solveBeta, gen_beta_est

include("ADVI/ADVI.jl")
include("Priors.jl")
include("Constants.jl")
include("vae.jl")
include("State.jl")

include("compute_Z.jl")
include("prob_miss.jl")
include("loglike.jl")
include("logprior.jl")
include("logq.jl")
include("compute_elbo.jl")

include("fit.jl")

end # VB
