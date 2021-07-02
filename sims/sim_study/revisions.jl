# NOTE: I need this for revisions. Fortunately, I can still use BSON.parse, though a little tedious.
#
include("revisions-util.jl")
using DelimitedFiles

# Path to MCMC output.
path_to_small_sim_results = "results/sim-paper/sim_Nfac500_KMCMC05/output.bson"
path_to_large_sim_results = "results/sim-paper/sim_Nfac5000_KMCMC10/output.bson"
# path_to_cb_results = nothing

# Get posterior for Z and lambda.
small_sim_summary = get_Z_lam(path_to_small_sim_results)
large_sim_summary = get_Z_lam(path_to_large_sim_results)
# cb_summary = nothing

# Write results.
outdir = "revision-summaries"
write_summary("$(outdir)/sim-small-fam-mcmc", small_sim_summary)
write_summary("$(outdir)/sim-large-fam-mcmc", large_sim_summary)
# write_summary("$(outdir)/cb-fam-mcmc", cb_summary)

