module Settings
# NOTE: Modify these things!
simname = "test-sim"
results_dir_prefix = "results/$(simname)" # relative to `parallel_sim.jl`
aws_bucket_prefix = "s3://cytof-repfam/$(simname)"

function results_dir(scale, kmcmc, seed_data)
  "$(results_dir_prefix)/seed_$(seed_data)/scale_$(scale)/Kmcmc_$(kmcmc)"
end

settings = [Dict(:simname => simname,
                 :repfamdistscale => scale,
                 :Kmcmc => kmcmc,
                 :seed_data => seed_data,
                 :seed_mcmc => 0,
                 :results_dir => results_dir(scale, kmcmc, seed_data))
            for scale in [0, 10],
            kmcmc in [2, 3, 4, 5, 6, 7, 15],
            seed_data in [1:5]]

settings = vec(settings)
end
