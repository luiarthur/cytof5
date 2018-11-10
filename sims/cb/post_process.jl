include("post_process_defs.jl")

OUTDIR = length(ARGS) > 0 ? ARGS[1] : "results/K_MCMC20_L_MCMC5_scale0.1_SEED0/"

post_process(OUTDIR)
