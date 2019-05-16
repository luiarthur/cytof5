#!/bin/bash

# Source a utility function
source ../sim_study/engine.sh

# Results directory
RESULTS_DIR=$1

# AWS Bucket to store results
AWS_BUCKET=$2

# Maximum number of cores to use
MAX_CORES=10

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
SEED=0
SUBSAMPLE="1.0"
isTest=""
otherFlags=""
SMARTINIT="true"
YQUANTILES=("0 .2 .4" "0 .15 .3")


for m in `seq ${#YQUANTILES[@]}`; do
  # Experiment name
  EXP_NAME="K_MCMC${K_MCMC}_missmech${m}"

  # Julia Command to run
  jlCmd="julia cb.jl --K_MCMC=${k_mcmc} \
    --L0_MCMC=${L0_MCMC} --L1_MCMC=${L1_MCMC} \
    --subsample=$SUBSAMPLE \
    --tau0=$TAU0 --tau1=$TAU1 \
    --RESULTS_DIR=$RESULTS_DIR --EXP_NAME=$EXP_NAME \
    --MCMC_ITER=$MCMC_ITER --BURN=$BURN --SEED=${SEED} \
    --smartinit=$SMARTINIT \
    --dnoisy=${DNOISY} \
    --yQuantiles='${YQUANTILES[$((m-1))]}' \
    --DATA_PATH=${DATA_PATH} \
    ${otherFlags}"

  engine $RESULTS_DIR $AWS_BUCKET $EXP_NAME "$jlCmd" $MAX_CORES $STAGGER_TIME
done
