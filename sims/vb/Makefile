SHELL:=/bin/bash
.PHONY: cleansim sim retrieveall retrieve

# Simulation data experiment name
SIM_EXP_NAME = vb-sim-paper

# Simulation results dir
SIM_RESULTS_DIR = results/$(SIM_EXP_NAME)/

# Sim AWS bukceT
SIM_AWS_BUCKET = s3://cytof-vb/$(SIM_EXP_NAME)

### Sim Make cmds ###
sim:
	bash sim.sh $(SIM_RESULTS_DIR)/ $(SIM_AWS_BUCKET)

cleansim:
	rm -rf $(SIM_RESULTS_DIR)/*

retrieve:
	aws s3 sync $(SIM_AWS_BUCKET) $(SIM_RESULTS_DIR) --exclude "*/out.bson"

retrieveall:
	aws s3 sync $(SIM_AWS_BUCKET) $(SIM_RESULTS_DIR)