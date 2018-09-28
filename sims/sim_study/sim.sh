#!/bin/bash

# Maximum number of cores to use
MAX_CORES=27

# AWS Bucket to store results
AWS_BUCKET="s3://cytof-sim-results"

# Experiment settings
I=3
J=32
N_factor="100 1000 10000"
K=8
L=4
K_MCMC=10
L_MCMC=5
b0PriorSd="0.1 1 10"
b1PriorScale="0.1 1 10"
RESULTS_DIR="results/"
SEED=0

# simulation number, just for book keeping. Ignore this.
simNumber=0


for nFac in $N_factor; do
  for b0Sd in $b0PriorSd; do
    for b1Scale in $b1PriorScale; do
      # Simulation number
      simNumber=$((simNumber + 1)) 

      # Experiment name
      exp_name="I${I}_J${J}_N_factor${nFac}_K${K}_L${L}_K_MCMC${K_MCMC}_L_MCMC${L_MCMC}_b0PriorSd${b0Sd}_b1PriorScale${b1Scale}_SEED${SEED}"

      # Output directory
      outdir="$RESULTS_DIR/$exp_name/"
      mkdir -p $outdir

      # Julia Command to run
      jlCmd="julia sim.jl --I=${I} --J=${J} --N_factor=${nFac} --K=${K} --L=${L} --K_MCMC=${K_MCMC} --L_MCMC=${L_MCMC} --b0PriorSd=${b0Sd} --b1PriorScale=${b1Scale} --SEED=${SEED} --RESULTS_DIR=$RESULTS_DIR --EXP_NAME=$exp_name"

      # Sync results to S3
      syncToS3="aws s3 sync $RESULTS_DIR $AWS_BUCKET"

      # Remove output files to save space on cluster
      rmOutput="rm -rf ${outdir}"

      cmd="$jlCmd > ${outdir}/log.txt && $syncToS3 && $rmOutput"

      sem -j $MAX_CORES $cmd
      echo $cmd

      echo "Results for simulation $simNumber -> $outdir"
    done
  done
done
