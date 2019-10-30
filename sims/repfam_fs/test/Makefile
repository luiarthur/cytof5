SHELL:=/bin/bash

.PHONY: cleansim sim retrieveallsim retrievesim postprocsim metrics

# Simulation data experiment name
# SIM_EXP_NAME = test-sims
SIM_EXP_NAME = test-sims-1

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
	aws s3 sync $(SIM_RESULTS_DIR) $(SIM_AWS_BUCKET) --exclude '*.nfs'

metrics:
	python3 metrics.py $(SIM_RESULTS_DIR)

notebook:
	jupyter notebook
