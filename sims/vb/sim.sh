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

# SIMDATA PATH
SIMDAT_PATH=(
  [5]="../sim_study/simdata/kills-flowsom/N500/K5/90/simdat.bson"
  [10]="../sim_study/simdata/kills-flowsom/N5000/K10/1/simdat.bson"
)

# K
K="5 10"

# Seeds to use
SEEDS=`seq -w 10`

# TODO
BATCHSIZES="500 1000 2000"
K_VB="5 10 30"

for k_vb in $K_VB; do
  for bs in $BATCHSIZES; do
    for k in $K; do
      for seed in $SEEDS; do
        # Experiment name
        EXP_NAME=BS${bs}/K_VB${k_vb}/K${k}/$seed
        # EXP_NAME=K${k}/BS${bs}/K_VB${k_vb}/$seed # TODO: use this instead

        # Dir for experiment results
        EXP_DIR=$RESULTS_DIR/$EXP_NAME/
        mkdir -p $EXP_DIR
        echo $EXP_DIR

        # julia command to run
        jlCmd="julia vb_sim.jl $seed $EXP_DIR ${SIMDAT_PATH[$k]} $k_vb $bs"

        engine $RESULTS_DIR $AWS_BUCKET $EXP_NAME "$jlCmd" $MAX_CORES $STAGGER_TIME
      done
    done
  done
done
