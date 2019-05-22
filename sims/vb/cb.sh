#!/bin/bash

# Load parallelizer script
source ../sim_study/engine.sh

# Results directory
RESULTS_DIR=$1

# AWS Bucket to store results
AWS_BUCKET=$2

# Maximum number of cores to use
MAX_CORES=15

# STAGGER_TIME in seconds. To avoid mass dumping to disk simultaneously. 
STAGGER_TIME=0

# Path to CB data
DATA_PATH="../cb/data/cytof_cb_float32.bson"

# Seeds to use
SEEDS=`seq -w 10`

# TODO
BATCHSIZES="100 500 2000"
K_VB="30 10 5"

for bs in $BATCHSIZES; do
  for k_vb in $K_VB; do
    for seed in $SEEDS; do
      # Experiment name
      EXP_NAME=BS${bs}/K_VB${k_vb}/$seed

      # Dir for experiment results
      EXP_DIR=$RESULTS_DIR/$EXP_NAME/
      mkdir -p $EXP_DIR
      echo $EXP_DIR

      # julia command to run
      jlCmd="julia vb_cb.jl $seed $EXP_DIR $DATA_PATH $k_vb $bs"

      engine $RESULTS_DIR $AWS_BUCKET $EXP_NAME "$jlCmd" $MAX_CORES $STAGGER_TIME
    done
  done
done
