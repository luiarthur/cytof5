# NOTE: Modify these things!
simname = "test-sim"

function outdir_suffix(scale, kmcmc, seed_data)
  return "seed_$(seed_data)/scale_$(scale)/Kmcmc_$(kmcmc)"
end

settings = [Dict(:simname => simname,
                 :repfam_dist_scale => scale,
                 :Kmcmc => kmcmc,
                 :seed_data => seed_data,
                 :seed_mcmc => 0,
                 :outdir_suffix => outdir_suffix(scale, kmcmc, seed_data))
            for scale in [0, 10],
            kmcmc in [2, 3, 5],
            seed_data in 2:3]

settings = vec(settings)
