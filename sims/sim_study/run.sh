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
I=3
J=32
L0_MCMC=5
L1_MCMC=5
SEED=0
MCMC_ITER=6000
BURN=10000
K=(5 10)
N_factor=(500 5000)

declare -A K_MCMC_GROUP
K_MCMC_GROUP=( [0]="`seq -w 2 10`" [1]="`seq -w 2 2 20`")

if [[ $@ == **--test** ]]
then
  MCMC_ITER=20
  BURN=10
  N_factor=(50 100)
  K_MCMC_GROUP=( [0]="`seq -w 2 3`" [1]="`seq -w 2 2 4`")
fi


for simNum in `seq 0 1`; do
  n_factor=${N_factor[$simNum]}
  k=${K[$simNum]}
  K_MCMC=${K_MCMC_GROUP[$simNum]}

  for k_mcmc in ${K_MCMC}; do
    # EXPERIMENT NAME
    EXP_NAME="sim_Nfac${n_factor}_K${k}_KMCMC${k_mcmc}"

    # DIRECTORY FOR EXPERIMENT RESULTS
    EXP_DIR=$RESULTS_DIR/$EXP_NAME/
    mkdir -p $EXP_DIR

    # Julia Command to run
    jlCmd="julia sim.jl\
      --I=${I} \
      --J=${J} \
      --L0_MCMC=${L0_MCMC} \
      --L1_MCMC=${L1_MCMC} \
      --N_factor=${n_factor} \
      --K=${k} \
      --K_MCMC=${k_mcmc} \
      --RESULTS_DIR=$RESULTS_DIR \
      --EXP_NAME=$EXP_NAME \
      --MCMC_ITER=$MCMC_ITER \
      --BURN=$BURN \
      --SEED=${SEED}"

    engine $RESULTS_DIR $AWS_BUCKET $EXP_NAME "$jlCmd" $MAX_CORES $STAGGER_TIME
  done
done

