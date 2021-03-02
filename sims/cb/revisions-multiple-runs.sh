#!/bin/bash

# Source a utility function
source ../sim_study/engine.sh

# Results directory
RESULTS_DIR=$1

# AWS Bucket to store results
AWS_BUCKET=$2

# Maximum number of cores to use
MAX_CORES=5

# STAGGER_TIME in seconds. To avoid mass dumping to disk simultaneously. 
STAGGER_TIME=100

# Experiment settings
MCMC_ITER=6000
BURN=10000
K_MCMC="21"
DNOISY="normal"
NOISY_SCALE="3.16"
L0_MCMC="5"
L1_MCMC="3"
TAU0="10.0"
TAU1="10.0"
DATA_PATH="data/cytof_cb_with_nan.jld2"
SEED=`seq 0 4`
SUBSAMPLE="1.0"
isTest=""
otherFlags=""
SMARTINIT="true"
YQUANTILES="0.0 .25 .5"


for seed in `seq ${SEED}`; do
  # Experiment name
  EXP_NAME="revision_K_MCMC${K_MCMC}_seed${seed}"

  # Julia Command to run
  jlCmd="julia cb.jl --K_MCMC=${K_MCMC} \
    --L0_MCMC=${L0_MCMC} --L1_MCMC=${L1_MCMC} \
    --subsample=$SUBSAMPLE \
    --tau0=$TAU0 --tau1=$TAU1 \
    --RESULTS_DIR=$RESULTS_DIR --EXP_NAME=$EXP_NAME \
    --MCMC_ITER=$MCMC_ITER --BURN=$BURN --SEED=${seed} \
    --smartinit=$SMARTINIT \
    --dnoisy=${DNOISY} \
    --yQuantiles='${YQUANTILES}' \
    --DATA_PATH=${DATA_PATH} \
    ${otherFlags}"

  engine $RESULTS_DIR $AWS_BUCKET $EXP_NAME "$jlCmd" $MAX_CORES $STAGGER_TIME
done
