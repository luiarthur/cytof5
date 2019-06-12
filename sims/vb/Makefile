SHELL:=/bin/bash
.PHONY: cleansim sim retrieveallsim retrievesim postprocsim
.PHONY: cleancb cb retrieveallcb retrievecb postproccb

# Simulation data experiment name
SIM_EXP_NAME = vb-sim-paper
# Simulation results dir
SIM_RESULTS_DIR = results/$(SIM_EXP_NAME)/
# Sim AWS bukceT
SIM_AWS_BUCKET = s3://cytof-vb/$(SIM_EXP_NAME)


# CB data experiment name
CB_EXP_NAME = vb-cb-paper
# CB results dir
CB_RESULTS_DIR = results/$(CB_EXP_NAME)/
# CB AWS bukceT
CB_AWS_BUCKET = s3://cytof-vb/$(CB_EXP_NAME)


### Sim Make cmds ###
sim:
	bash sim.sh $(SIM_RESULTS_DIR)/ $(SIM_AWS_BUCKET)

cleansim:
	rm -rf $(SIM_RESULTS_DIR)/*

retrievesim:
	aws s3 sync $(SIM_AWS_BUCKET) $(SIM_RESULTS_DIR) --exclude "*/output.bson"

retrieveallsim:
	aws s3 sync $(SIM_AWS_BUCKET) $(SIM_RESULTS_DIR)

postprocsim:
	julia post_process.jl $(SIM_RESULTS_DIR)

sendsim:
	aws s3 sync $(SIM_RESULTS_DIR) $(SIM_AWS_BUCKET) --exclude '*.nfs'

hyperparamsim:
	python3 hyper_param_visual.py $(SIM_RESULTS_DIR)

getBestSIM:
	aws s3 sync   $(SIM_AWS_BUCKET)/K10/BS2000/K_VB30/10/   $(SIM_RESULTS_DIR)/K10/BS2000/K_VB30/10/
	aws s3 sync   $(SIM_AWS_BUCKET)/K5/BS2000/K_VB30/10/    $(SIM_RESULTS_DIR)/K5/BS2000/K_VB30/10/

sendBestSIM:
	aws s3 sync   $(SIM_RESULTS_DIR)/K10/BS2000/K_VB30/10/  $(SIM_AWS_BUCKET)/K10/BS2000/K_VB30/10/   
	aws s3 sync   $(SIM_RESULTS_DIR)/K5/BS2000/K_VB30/10/   $(SIM_AWS_BUCKET)/K5/BS2000/K_VB30/10/    

### CB Make cmds ###
cb:
	bash cb.sh $(CB_RESULTS_DIR)/ $(CB_AWS_BUCKET)

cleancb:
	rm -rf $(CB_RESULTS_DIR)/*

retrievecb:
	aws s3 sync $(CB_AWS_BUCKET) $(CB_RESULTS_DIR) --exclude "*/output.bson"

retrieveallcb:
	aws s3 sync $(CB_AWS_BUCKET) $(CB_RESULTS_DIR)

postproccb:
	julia post_process.jl $(CB_RESULTS_DIR)

sendcb:
	aws s3 sync $(CB_RESULTS_DIR) $(CB_AWS_BUCKET) --exclude '*.nfs'

hyperparamcb:
	python3 hyper_param_visual.py $(CB_RESULTS_DIR)

getBestCB:
	aws s3 sync   $(CB_AWS_BUCKET)/BS2000/K_VB30/09/    $(CB_RESULTS_DIR)/BS2000/K_VB30/09/

sendBestCB:
	aws s3 sync   $(CB_RESULTS_DIR)/BS2000/K_VB30/09/   $(CB_AWS_BUCKET)/BS2000/K_VB30/09/  

sendTestCB:
	aws s3 sync   results/test/vb-cb-paper/   $(CB_AWS_BUCKET)/test/  
