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
SIM_EXP_NAME = test-sims-5-7

# Current experiment 
# Using Z3, which has several similar columns
# USing sig2_TRUE = 0.5
# Using prior: W*_{ik} ~ Gamma(10, 1)
# Keep 3000 samples and burn 10000 for better LPML
# Use sample size of 500
SIM_EXP_NAME = test-sims-5-8

### End of Simulation data experiment name ###

# Simulation results dir
SIM_RESULTS_DIR = results/$(SIM_EXP_NAME)

# Sim AWS bukceT
SIM_AWS_BUCKET = s3://cytof-repfam/$(SIM_EXP_NAME)


### Sim Make cmds ###
sim:
	bash sim.sh $(SIM_RESULTS_DIR)/ $(SIM_AWS_BUCKET)

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

