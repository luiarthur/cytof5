#!/bin/bash

# Load parallelizer script
source ../../sim_study/engine.sh

# Results directory
RESULTS_DIR=$1

# AWS Bucket to store results
AWS_BUCKET=$2

# Maximum number of cores to use
MAX_CORES=24

# STAGGER_TIME in seconds. To avoid mass dumping to disk simultaneously. 
STAGGER_TIME=0

# SETTINGS
# KMCMC=`seq 2 5`
# Z_idx="1 2"
# repfamdistscale="0 0.01 0.1 1 10"
# repfamdistscale="0 1 10"
KMCMC="03 04 05 06 07 15"
Z_idx="3"
repfamdistscale="0 10"
SEED=`seq 0 4`

for seed in $SEED; do
  for kmcmc in $KMCMC; do
    for zidx in $Z_idx; do
      for scale in $repfamdistscale; do
        # Experiment name
        EXP_NAME=KMCMC${kmcmc}/z${zidx}/scale${scale}/seed${seed}

        # Dir for experiment results
        EXP_DIR=$RESULTS_DIR/$EXP_NAME/
        mkdir -p $EXP_DIR
        echo $EXP_DIR

        # julia command to run
        jlCmd="julia small_sim.jl $EXP_DIR $scale $kmcmc $zidx $seed"

        engine $RESULTS_DIR $AWS_BUCKET $EXP_NAME "$jlCmd" $MAX_CORES $STAGGER_TIME
      done
    done
  done
done
