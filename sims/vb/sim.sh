#!/bin/bash

# Load parallelizer script
source ../sim_study/engine.sh

# Results directory
RESULTS_DIR=$1

# AWS Bucket to store results
AWS_BUCKET=$2

# Maximum number of cores to use
MAX_CORES=10

# STAGGER_TIME in seconds. To avoid mass dumping to disk simultaneously. 
STAGGER_TIME=0

# Seeds to use
SEEDS=`seq -w 10`

for seed in $SEEDS; do
  # Experiment name
  EXP_NAME=$seed

  # Dir for experiment results
  EXP_DIR=$RESULTS_DIR/$EXP_NAME/
  mkdir -p $EXP_DIR

  # julia command to run
  jlCmd="julia vb-sim.jl $seed $EXP_DIR"

  engine $RESULTS_DIR $AWS_BUCKET $EXP_NAME "$jlCmd" $MAX_CORES $STAGGER_TIME
done
