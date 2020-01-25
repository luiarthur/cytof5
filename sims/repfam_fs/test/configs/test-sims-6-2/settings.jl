# Simulation Name
simname = basename(pwd())

# NOTE: write to scratchdir
function outdir_suffix(scale, kmcmc, seed_data)
  return "seed_$(seed_data)/scale_$(scale)/Kmcmc_$(kmcmc)"
end

# NOTE: modify
settings = [Dict(:simname => simname,
                 :repfam_dist_scale => scale,
                 :N => [300, 300],
                 :Z_idx => 4,  # J = 20
                 :thin_samps => 2,
                 :nburn => 10000,
                 :nsamps => 2000,
                 # :nburn => 10, # NOTE: test
                 # :nsamps => 20, # NOTE: test
                 :Lmcmc => Dict(0 => 2, 1 => 2),
                 :Kmcmc => kmcmc,
                 :seed_data => seed_data,
                 :seed_mcmc => 0,
                 :outdir_suffix => outdir_suffix(scale, kmcmc, seed_data))
            for scale in [0, 10],
            kmcmc in [3, 4, 5, 6, 7, 15],
            seed_data in 1:5]

settings = vec(settings)
