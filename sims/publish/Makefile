SHELL := /bin/bash

.PHONY: get send

SCRATCH_DIR = /scratchdata/alui2/cytof
RESULTS_DIR = results
PAPER_AWS_BUCKET = s3://cytof-fam-paper/img/post/

# CB
CB_AWS_BUCKET = s3://cytof-cb-results/cb-paper
CB_BEST_AWS_BUCKET = $(CB_AWS_BUCKET)/K_MCMC21_L0_MCMC5_L1_MCMC3_tau0_10.0_tau1_10.0_SEED0_smartinittrue_subsample1.0_dnoisynormal_noisy_scale3.16
CB_MM1_AWS_BUCKET = $(CB_AWS_BUCKET)/K_MCMC21_missmech1
CB_MM2_AWS_BUCKET = $(CB_AWS_BUCKET)/K_MCMC21_missmech2

# Small sim
SIM_AWS_BUCKET = s3://cytof-sim-results/sim-paper
SIM_SMALL_K5_AWS_BUCKET  = $(SIM_AWS_BUCKET)/sim_Nfac500_KMCMC05
SIM_SMALL_MM1_AWS_BUCKET = $(SIM_AWS_BUCKET)/sim_Nfac500_missmech1
SIM_SMALL_MM2_AWS_BUCKET = $(SIM_AWS_BUCKET)/sim_Nfac500_missmech2

# Small big
SIM_BIG_K5_AWS_BUCKET  = SIM_AWS_BUCKET/sim_Nfac5000_KMCMC10
SIM_BIG_MM1_AWS_BUCKET = SIM_AWS_BUCKET/sim_Nfac5000_missmech1
SIM_BIG_MM2_AWS_BUCKET = SIM_AWS_BUCKET/sim_Nfac5000_missmech2

CB_DIR = results/cb
SIM_DIR = results/sim

# TAGS
TAGS = --exclude "*" --include "*.bson" --include "*.jld2" --include "*.jld2" --include "*log.txt"

### MAIN ###
getCB:
	aws s3 sync $(CB_BEST_AWS_BUCKET) $(CB_DIR)/best $(TAGS)
	aws s3 sync $(CB_MM1_AWS_BUCKET)  $(CB_DIR)/mm1  $(TAGS)
	aws s3 sync $(CB_MM2_AWS_BUCKET)  $(CB_DIR)/mm2  $(TAGS)


getSmallSim:
	aws s3 sync $(SIM_SMALL_K5_AWS_BUCKET)   $(SIM_DIR)/small/best $(TAGS)
	aws s3 sync $(SIM_SMALL_MM1_AWS_BUCKET)  $(SIM_DIR)/small/mm1  $(TAGS)
	aws s3 sync $(SIM_SMALL_MM2_AWS_BUCKET)  $(SIM_DIR)/small/mm2  $(TAGS)


getBigSim:
	aws s3 sync $(SIM_BIG_K5_AWS_BUCKET)   $(SIM_DIR)/big/best $(TAGS)
	aws s3 sync $(SIM_BIG_MM1_AWS_BUCKET)  $(SIM_DIR)/big/mm1  $(TAGS)
	aws s3 sync $(SIM_BIG_MM2_AWS_BUCKET)  $(SIM_DIR)/big/mm2  $(TAGS)


send: 
	aws s3 sync $(RESULTS_DIR)  $(PAPER_AWS_BUCKET)


get: getCB getSmallSim getBigSim
	aws s3 sync $(CB_AWS_BUCKET)   $(CB_DIR)/logs  --exclude "*" --include "*log.txt"
	aws s3 sync $(SIM_AWS_BUCKET)  $(SIM_DIR)/logs --exclude "*" --include "*log.txt"

get2scratch:
	mkdir -p $(SCRATCH_DIR)/results
	cp Makefile $(SCRATCH_DIR)
	cd $(SCRATCH_DIR)/ && make get
	# ln -s $(SCRATCH_DIR)/results results


