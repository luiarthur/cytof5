#!/bin/bash

# Source a utility function
source engine.sh

# Results directory
RESULTS_DIR=$1

# AWS Bucket to store results
AWS_BUCKET=$2

# Maximum number of cores to use
MAX_CORES=20

# STAGGER_TIME in seconds. To avoid mass dumping to disk simultaneously. 
STAGGER_TIME=100

# Experiment Settings
L0_MCMC=5
L1_MCMC=5
MCMC_ITER=6000
BURN=10000

# PATH TO SIMULATED DATA DIR
SIMDAT_DIR="simdata/kills-flowsom/"

# Create dictionary key
N_factor=(500 5000)

# Create yQuantiles
YQUANTILES=("0 .2 .4" "0 .15 .3")

# Create pBounds
PBOUNDS=".05 .8 .05"

# Create SEED dict
declare -A SEED
SEED=( [${N_factor[0]}]=90 [${N_factor[1]}]=1)

# Create K dict
declare -A K
K=( [${N_factor[0]}]="5" [${N_factor[1]}]="10")


if [[ $@ == **--test** ]]
then
  MCMC_ITER=20
  BURN=10
  N_factor=(50 100)
fi

# MAIN
for n_factor in ${N_factor[@]}; do
  seed=${SEED[$n_factor]}
  k=${K[$n_factor]}
  simdat_path="${SIMDAT_DIR}/N${n_factor}/K${k}/${seed}/simdat.bson"

  for m in `seq ${#YQUANTILES[@]}`; do
    # EXPERIMENT NAME
    EXP_NAME="sim_Nfac${n_factor}_missmech${m}"

    # DIRECTORY FOR EXPERIMENT RESULTS
    EXP_DIR=$RESULTS_DIR/$EXP_NAME/
    mkdir -p $EXP_DIR

    # Julia Command to run
    jlCmd="julia sim.jl\
      --simdat_path=${simdat_path} \
      --L0_MCMC=${L0_MCMC} \
      --L1_MCMC=${L1_MCMC} \
      --K_MCMC=${k} \
      --yQuantiles='${YQUANTILES[$((m-1))]}' \
      --pBounds='${PBOUNDS}' \
      --RESULTS_DIR=$RESULTS_DIR \
      --EXP_NAME=$EXP_NAME \
      --MCMC_ITER=$MCMC_ITER \
      --BURN=$BURN \
      --SEED=${seed}"

    engine $RESULTS_DIR $AWS_BUCKET $EXP_NAME "$jlCmd" $MAX_CORES $STAGGER_TIME
    # echo $jlCmd
  done
done

