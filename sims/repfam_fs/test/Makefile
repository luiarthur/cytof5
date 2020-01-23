SHELL:=/bin/bash

.PHONY: cleansim sim retrieveallsim retrievesim postprocsim metrics killsim

### Simulation data experiment name ###
# Initial test
# SIM_EXP_NAME = test-sims

# all experiments took two components
# SIM_EXP_NAME = test-sims-1

# Basically point mass prior mean on at 1 for deltaz
# SIM_EXP_NAME = test-sims-2

# deltaz ~ TN(0, 1, .75, Inf)
# SIM_EXP_NAME = test-sims-3

# deltaz ~ TN(0, .1, 0, Inf)
# sig2 ~ IG(11, 5)
# Z_thin = J and flip one bit each time -> update one bit in each row, row by row.
# update each r_ik 
# plot dden correctly
# scale = (0, 1, 10)
# NMCMC = 1000
# NBURN = 1000
# SIM_EXP_NAME = test-sims-4

# Using Z3, which has several similar columns
# SIM_EXP_NAME = test-sims-5

# Using Z3, which has several similar columns
# USing sig2_TRUE = 0.5
# SIM_EXP_NAME = test-sims-5-5

# Using Z3, which has several similar columns
# USing sig2_TRUE = 0.5
# Using prior: W*_{ik} ~ Gamma(10, 1)
# SIM_EXP_NAME = test-sims-5-6

# Using Z3, which has several similar columns
# USing sig2_TRUE = 0.5
# Using prior: W*_{ik} ~ Gamma(10, 1)
# Keep 3000 samples and burn 10000 for better LPML
# Except for the last point, this is same as 5-6.
# SIM_EXP_NAME = test-sims-5-7

# Using Z3, which has several similar columns
# USing sig2_TRUE = 0.5
# Using prior: W*_{ik} ~ Gamma(10, 1)
# Keep 3000 samples and burn 10000 for better LPML
# Use sample size of 500
# SIM_EXP_NAME = test-sims-5-8

# Using Z4, which has several similar columns and J=20.
# USing sig2_TRUE = 0.5
# Using prior: W*_{ik} ~ Gamma(10, 1)
# 2000 sampels (thinned by 2), burn 10000.
# Use sample size of 300
# NOTE: To return to sims-5-6, 
# sim.sh:
#   - Z_idx=3
# Makefile:
#   = SIM_EXP_NAME = test-sims-5-6
# SIM_EXP_NAME = test-sims-5-9

# Using Z3, which has several similar columns
# USing sig2_TRUE = 0.5
# Using prior: W*_{ik} ~ Gamma(10, 1)
# Keep 2000 samples, thinned by 2, and burn 10000 for better LPML
# Use sample size of 1000
# SIM_EXP_NAME = test-sims-5-10

# Using Z3, which has several similar columns
# USing sig2_TRUE = 0.5
# Using prior: W*_{ik} ~ Gamma(10, 1)
# Keep 3000 samples and burn 10000 for better LPML
# Using loglike = p(y, m | theta) which doesn't drop p(y|theta) for missing data.
# Except for the last point, this is the same as 5-7. 
# SIM_EXP_NAME = test-sims-5-11

# Using Z3, which has several similar columns
# USing sig2_TRUE = 0.5
# Using prior: W*_{ik} ~ Gamma(10, 1)
# Keep 3000 samples and burn 10000 for better LPML
# Using loglike = p(y, m | theta) which doesn't drop p(y|theta) for missing data.
# Does for one seed (0), but 5 random MCMC seeds.
# N=[1000, 1000]
# Except for the last point, this is the same as 5-11. 
# SIM_EXP_NAME = test-sims-5-12

# Using Z3, which has several similar columns
# USing sig2_TRUE = 0.5
# Using prior: W*_{ik} ~ Gamma(10, 1)
# Keep 3000 samples and burn 10000 for better LPML
# Using loglike = p(y, m | theta) which doesn't drop p(y|theta) for missing data.
# SIM_EXP_NAME = test-sims-6-1

# Current experiment 
# with Z4 (J=20)
# N=[300, 300]
SIM_EXP_NAME = test-sims-6-2

# Next experiment 
# with Z4 (J=20)
# N=[1000, 1000]
# SIM_EXP_NAME = test-sims-6-3


### End of Simulation data experiment name ###

# Simulation settings dir
SIM_CONFIG_DIR = configs/$(SIM_EXP_NAME)

# Simulation results dir
SIM_RESULTS_DIR = results/$(SIM_EXP_NAME)

# Sim AWS bukceT
SIM_AWS_BUCKET = s3://cytof-repfam/$(SIM_EXP_NAME)

### Sim Make cmds ###
sim:
	@# bash sim.sh $(SIM_RESULTS_DIR)/ $(SIM_AWS_BUCKET)
	@echo "Printing results to results/$(SIM_EXP_NAME)-master-log.txt"
	julia parsim.jl $(SIM_CONFIG_DIR) &> results/$(SIM_EXP_NAME)-master-log.txt &

cleansim:
	rm -rf $(SIM_RESULTS_DIR)/*

retrievesim:
	aws s3 sync $(SIM_AWS_BUCKET) $(SIM_RESULTS_DIR) \
		--exclude "*/output.bson" --exclude "*/simdat.bson"

retrieveallsim:
	aws s3 sync $(SIM_AWS_BUCKET) $(SIM_RESULTS_DIR)

postprocsim:
	julia post_process.jl $(SIM_RESULTS_DIR) $(SIM_AWS_BUCKET)

sendsim:
	aws s3 sync $(SIM_RESULTS_DIR) $(SIM_AWS_BUCKET) --exclude '*.nfs*'

metrics:
	python3 metrics.py $(SIM_RESULTS_DIR)
	python3 rprob.py $(SIM_RESULTS_DIR)

notebook:
	jupyter notebook

killsim:
	pids=`grep -r pid $(SIM_RESULTS_DIR) | grep -oP '(?<=pid).*' | grep -oP '\d+'`; \
			 for p in $$pids; do kill $$p; done

