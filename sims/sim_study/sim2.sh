#!/bin/bash

# Maximum number of cores to use
MAX_CORES=27

# AWS Bucket to store results
AWS_BUCKET="s3://cytof-sim2-results"

# STAGGER_TIME in seconds. To avoid mass dumping to disk simultaneously. 
STAGGER_TIME=100

# Experiment settings
I=3
J=32
N_factor="100"
K=8
L=4
K_MCMC="6 7 8 9 10 11"
L_MCMC=5
betaPriorScale="0.01"
RESULTS_DIR="results/sim2/"
SEED=0

# simulation number, just for book keeping. Ignore this.
simNumber=0


for nFac in $N_factor; do
  for bs in $betaPriorScale; do
    for kmcmc in $K_MCMC; do
      # Simulation number
      simNumber=$((simNumber + 1)) 

      # Experiment name
      exp_name="I${I}_J${J}_N_factor${nFac}_K${K}_L${L}_K_MCMC${kmcmc}_L_MCMC${L_MCMC}_b0PriorSd${bs}_b1PriorScale${bs}_SEED${SEED}"

      # Output directory
      outdir="$RESULTS_DIR/$exp_name/"
      mkdir -p $outdir

      # Julia Command to run
      jlCmd="julia sim.jl --I=${I} --J=${J} --N_factor=${nFac} --K=${K} --L=${L} --K_MCMC=${kmcmc} --L_MCMC=${L_MCMC} --b0PriorSd=${bs} --b1PriorScale=${bs} --SEED=${SEED} --RESULTS_DIR=$RESULTS_DIR --EXP_NAME=$exp_name"

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
