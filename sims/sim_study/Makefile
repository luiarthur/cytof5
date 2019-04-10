SHELL:=/bin/bash
.PHONY: test run clean retrieve retrieveAll metrics

# EXPERIMENT NAME
EXP_NAME = sim-paper

# RESULTS DIRECTORY
RESULTS_DIR = results/$(EXP_NAME)

# AWS BUCKET
AWS_BUCKET = s3://cytof-sim-results/$(subst _,-,$(EXP_NAME))

# TEST DIRECTORY
TEST_DIR = $(RESULTS_DIR)/test/

# OLD EXPERIMENTS
#{{{1
# EXP_NAME = NONE
#}}}1

run:
	bash run.sh  $(RESULTS_DIR)/  $(AWS_BUCKET)

test:
	bash run.sh  $(TEST_DIR)/  $(AWS_BUCKET)  --test

clean:
	rm -rf $(RESULTS_DIR)/*

retrieve: 
	aws s3 sync $(AWS_BUCKET) $(RESULTS_DIR) --exclude "*/output.bson"

retrieveAll:
	aws s3 sync $(AWS_BUCKET) $(RESULTS_DIR)

metrics:
	@echo TODO	
	# python3 parse_metrics.py results/$(EXP_NAME) elbo
	# python3 parse_metrics.py results/$(EXP_NAME) ll
	# python3 parse_metrics.py results/$(EXP_NAME) logpost
