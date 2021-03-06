SHELL:=/bin/bash
.PHONY: test run clean retrieve retrieveAll metrics flowsom getsimdat pullAndProcess missmechsen send

# EXPERIMENT NAME
EXP_NAME = sim-paper

# RESULTS DIRECTORY
RESULTS_DIR = results/$(EXP_NAME)

# AWS BUCKET
AWS_BUCKET = s3://cytof-sim-results/$(subst _,-,$(EXP_NAME))

# TEST DIRECTORY
TEST_DIR = $(RESULTS_DIR)/test/

# AWS BUCKET for simulated data
SIMDATA_AWS_BUCKET = s3://cytof-sim-results/simdata/

# DATA DIR
SIMDATA_DIR = simdata/

# OLD EXPERIMENTS
#{{{1
# EXP_NAME = sim-paper-flat# flatter noisy dist
# EXP_NAME = sim-paper-fixed-lpml
# EXP_NAME = sim-paper-fixed-lpml-2#includes p(y_inj | \theta) for missing y
# EXP_NAME = sim-paper-fixed-lpml-3# includes the previous fixes & correction to sig2
#}}}1

run:
	bash run.sh  $(RESULTS_DIR)/  $(AWS_BUCKET)

missmechsen:
	bash missmechsen.sh  $(RESULTS_DIR)/  $(AWS_BUCKET)

test:
	bash run.sh  $(TEST_DIR)/  $(AWS_BUCKET)  --test

clean:
	rm -rf $(RESULTS_DIR)/*

send: 
	aws s3 sync $(RESULTS_DIR) $(AWS_BUCKET) --exclude "*/output.bson"

sendmetrics:
	aws s3 sync $(RESULTS_DIR)/metrics $(AWS_BUCKET)/metrics/

retrieve: 
	aws s3 sync $(AWS_BUCKET) $(RESULTS_DIR) --exclude "*/output.bson"

retrieveAll:
	aws s3 sync $(AWS_BUCKET) $(RESULTS_DIR)

flowsom:
	julia flowsom.jl "simdata/kills-flowsom/N500/K5/90/simdat.bson"  "results/sim-paper/flowsom/N500/"
	julia flowsom.jl "simdata/kills-flowsom/N5000/K10/1/simdat.bson" "results/sim-paper/flowsom/N5000/"

getsimdat:
	aws s3 sync $(SIMDATA_AWS_BUCKET) $(SIMDATA_DIR)

metrics:
	python3 metrics.py $(RESULTS_DIR)
	./lpml_vs_small_clus.py

pullAndProcess:
	julia pullAndGenResults.jl $(AWS_BUCKET)
