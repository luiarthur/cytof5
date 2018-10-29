include("post_process_defs.jl")

OUTDIR = length(ARGS) > 0 ? ARGS[1] : "results/K_MCMC20_L_MCMC5_scale0.1_SEED0/"
DATAPATH = length(ARGS) > 1 ? ARGS[2] : "data/cytof_cb.jld2"

post_process(OUTDIR, DATAPATH)
