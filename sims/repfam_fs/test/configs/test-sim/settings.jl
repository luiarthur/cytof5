module Settings

# NOTE: Modify these things!
simname = "test-sim"
results_dir_prefix = "results/$(simname)" # relative to `parallel_sim.jl`
aws_bucket_prefix = "s3://cytof-repfam"
aws_bucket = "aws_bucket_prefix/$(simname)"

function results_dir(scale, kmcmc, seed_data)
  "$(results_dir_prefix)/seed_$(seed_data)/scale_$(scale)/Kmcmc_$(kmcmc)"
end

settings = [Dict(:simname => simname,
                 :repfam_dist_scale => scale,
                 :Kmcmc => kmcmc,
                 :seed_data => seed_data,
                 :seed_mcmc => 0,
                 :aws_bucket => aws_bucket,
                 :results_dir => results_dir(scale, kmcmc, seed_data))
            for scale in [0, 10],
            kmcmc in [2, 3, 5],
            seed_data in 2:3]

settings = vec(settings)

end
