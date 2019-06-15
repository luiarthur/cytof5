SHELL := /bin/bash

.PHONY: get send

# CB
CB_BEST_AWS_BUCKET = s3://cytof-cb-results/cb-paper/K_MCMC21_L0_MCMC5_L1_MCMC3_tau0_10.0_tau1_10.0_SEED0_smartinittrue_subsample1.0_dnoisynormal_noisy_scale3.16
CB_MM1_AWS_BUCKET = s3://cytof-cb-results/cb-paper/K_MCMC21_missmech1
CB_MM2_AWS_BUCKET = s3://cytof-cb-results/cb-paper/K_MCMC21_missmech2

# Small sim
SIM_SMALL_K5_AWS_BUCKET = s3://cytof-sim-results/sim-paper/sim_Nfac500_KMCMC05
SIM_SMALL_MM1_AWS_BUCKET = s3://cytof-sim-results/sim-paper/sim_Nfac500_missmech1
SIM_SMALL_MM2_AWS_BUCKET = s3://cytof-sim-results/sim-paper/sim_Nfac500_missmech2

# Small big
SIM_BIG_K5_AWS_BUCKET = s3://cytof-sim-results/sim-paper/sim_Nfac5000_KMCMC10
SIM_BIG_MM1_AWS_BUCKET = s3://cytof-sim-results/sim-paper/sim_Nfac5000_missmech1
SIM_BIG_MM2_AWS_BUCKET = s3://cytof-sim-results/sim-paper/sim_Nfac5000_missmech2

CB_DIR = results/cb
SIM_DIR = results/sim

# TAGS
TAGS = --exclude "*" --include "*.bson" --include "*.jld2" --include "*.jld2" --include "log.txt"

### MAIN ###
getCB:
	aws s3 sync $(CB_BEST_AWS_BUCKET) $(CB_DIR)/best $(TAGS)
	aws s3 sync $(CB_MM1_AWS_BUCKET)  $(CB_DIR)/mm1  $(TAGS)
	aws s3 sync $(CB_MM2_AWS_BUCKET)  $(CB_DIR)/mm2  $(TAGS)

sendCB:
	aws s3 sync $(CB_DIR)/best $(CB_BEST_AWS_BUCKET) 
	aws s3 sync $(CB_DIR)/mm1  $(CB_MM1_AWS_BUCKET)  
	aws s3 sync $(CB_DIR)/mm2  $(CB_MM2_AWS_BUCKET)  


getSmallSim:
	aws s3 sync $(SIM_SMALL_K5_AWS_BUCKET)   $(SIM_DIR)/small/best $(TAGS)
	aws s3 sync $(SIM_SMALL_MM1_AWS_BUCKET)  $(SIM_DIR)/small/mm1  $(TAGS)
	aws s3 sync $(SIM_SMALL_MM2_AWS_BUCKET)  $(SIM_DIR)/small/mm2  $(TAGS)

sendSmallim:
	aws s3 sync $(SIM_DIR)/small/best $(SIM_SMALL_K5_AWS_BUCKET)   
	aws s3 sync $(SIM_DIR)/small/mm1  $(SIM_SMALL_MM1_AWS_BUCKET)  
	aws s3 sync $(SIM_DIR)/small/mm2  $(SIM_SMALL_MM2_AWS_BUCKET)  


getBigSim:
	aws s3 sync $(SIM_BIG_K5_AWS_BUCKET)   $(SIM_DIR)/big/best $(TAGS)
	aws s3 sync $(SIM_BIG_MM1_AWS_BUCKET)  $(SIM_DIR)/big/mm1  $(TAGS)
	aws s3 sync $(SIM_BIG_MM2_AWS_BUCKET)  $(SIM_DIR)/big/mm2  $(TAGS)

sendBigSim:
	aws s3 sync $(SIM_DIR)/big/best $(SIM_BIG_K5_AWS_BUCKET)   
	aws s3 sync $(SIM_DIR)/big/mm1  $(SIM_BIG_MM1_AWS_BUCKET)  
	aws s3 sync $(SIM_DIR)/big/mm2  $(SIM_BIG_MM2_AWS_BUCKET)  


send: sendCB sendSmallSim sendBigSim

get: getCB getSmallSim getBigSim

