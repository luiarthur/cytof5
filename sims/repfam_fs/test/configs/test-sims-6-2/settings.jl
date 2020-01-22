module Settings

# Simulation Name
simname = "test-sims-6-2"  # NOTE: modify

# FIXME: write to scratchdir
results_dir_prefix = "results/$(simname)" # relative to `parallel_sim.jl`
aws_bucket_prefix = "s3://cytof-repfam/$(simname)"

function results_dir_suffix(scale, kmcmc, seed_data)
  return "seed_$(seed_data)/scale_$(scale)/Kmcmc_$(kmcmc)"
end

function results_dir(scale, kmcmc, seed_data)
  suffix = results_dir_suffix(scale, kmcmc, seed_data)
  return "$(results_dir_prefix)/$(suffix)"
end

function aws_bucket(scale, kmcmc, seed_data)
  suffix = results_dir_suffix(scale, kmcmc, seed_data)
  return "$(aws_bucket_prefix)/$(suffix)"
end

# NOTE: modify
settings = [Dict(:simname => simname,
                 :repfam_dist_scale => scale,
                 :N => [300, 300],
                 :Z_idx => 4,
                 :thin_samps => 2,
                 :nburn => 10000,
                 :nsamps => 2000,
                 # :nburn => 10, # NOTE: test
                 # :nsamps => 20, # NOTE: test
                 :Lmcmc => Dict(0 => 2, 1 => 2),
                 :Kmcmc => kmcmc,
                 :seed_data => seed_data,
                 :seed_mcmc => 0,
                 :aws_bucket => aws_bucket(scale, kmcmc, seed_data),
                 :results_dir => results_dir(scale, kmcmc, seed_data))
            for scale in [0, 10],
            kmcmc in [3, 4, 5, 6, 7, 15],
            seed_data in 1:5]

settings = vec(settings)

end
