#!/bin/bash

# Maximum number of cores to use
MAX_CORES=18

# AWS Bucket to store results
AWS_BUCKET="s3://cytof-sim-beta-tuner-init-results"

# STAGGER_TIME in seconds. To avoid mass dumping to disk simultaneously. 
STAGGER_TIME=100

# Experiment settings
MCMC_ITER=1000
BURN=10000
I=3
J=32
N_factor=100
K=8
L=4
K_MCMC=8
L_MCMC=5
betaPriorScale="0.01 0.1"
b0TunerInit="0.1 1.0 10.0"
b1TunerInit="0.1 1.0 10.0"
RESULTS_DIR="results/sim3/"
SEED=0

# simulation number, just for book keeping. Ignore this.
simNumber=0


for bs in $betaPriorScale; do
  for b0TI in $b0TunerInit; do
    for b1TI in $b1TunerInit; do
      # Simulation number
      simNumber=$((simNumber + 1)) 

      # Experiment name
      exp_name="I${I}_J${J}_N_factor${N_factor}_K${K}_L${L}_K_MCMC${K_MCMC}_L_MCMC${L_MCMC}_b0PriorSd${bs}_b1PriorScale${bs}_SEED${SEED}_b0TunerInit${b0TI}_b1TunerInit${b1TI}"

      # Output directory
      outdir="$RESULTS_DIR/$exp_name/"
      mkdir -p $outdir

      # Julia Command to run
      jlCmd="julia sim.jl --I=${I} --J=${J} --N_factor=${N_factor} --K=${K} \
        --L=${L} --K_MCMC=${K_MCMC} --L_MCMC=${L_MCMC} --b0PriorSd=${bs} \
        --b1PriorScale=${bs} --SEED=${SEED} --RESULTS_DIR=$RESULTS_DIR \
        --EXP_NAME=$exp_name --b0TunerInit=${b0TI} --b1TunerInit=${b1TI} \
        --MCMC_ITER=${MCMC_ITER} --BURN=${BURN}"

      # Sync results to S3
      syncToS3="aws s3 sync $RESULTS_DIR $AWS_BUCKET"

      # Remove output files to save space on cluster
      rmOutput="rm -rf ${outdir}"

      cmd="$jlCmd > ${outdir}/log.txt && $syncToS3 && $rmOutput"

      sem -j $MAX_CORES $cmd
      echo $cmd

      echo "Results for simulation $simNumber -> $outdir"

      sleep $STAGGER_TIME
    done
  done
done
