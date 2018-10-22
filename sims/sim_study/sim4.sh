#!/bin/bash

# Maximum number of cores to use
MAX_CORES=36

# AWS Bucket to store results
AWS_BUCKET="s3://cytof-vary-kmcmc-n1000"

# STAGGER_TIME in seconds. To avoid mass dumping to disk simultaneously. 
STAGGER_TIME=100

# Experiment settings
MCMC_ITER=1000
BURN=10000
I=3
J=32
N_factor="1000 10000"
K=8
L=4
K_MCMC="6 7 8 9 10"
L_MCMC=5
betaPriorScale="0.01 0.1 5.0"
betaTunerInit="0.1"
RESULTS_DIR="results/sim4/"
SEED="98 64"

# simulation number, just for book keeping. Ignore this.
simNumber=0


for nFac in $N_factor; do
  for bs in $betaPriorScale; do
    for k_mcmc in $K_MCMC; do
      # Simulation number
      simNumber=$((simNumber + 1)) 

      # Experiment name
      exp_name="I${I}_J${J}_N_factor${nFac}_K${K}_L${L}_K_MCMC${k_mcmc}_L_MCMC${L_MCMC}_betaPriorScale${bs}_betaTunerInit${betaTunerInit}_SEED${SEED}"

      # Output directory
      outdir="$RESULTS_DIR/$exp_name/"
      mkdir -p $outdir

      # Julia Command to run
      jlCmd="julia sim.jl --I=${I} --J=${J} --N_factor=${nFac} --K=${K} \
        --L=${L} --K_MCMC=${k_mcmc} --L_MCMC=${L_MCMC} --b0PriorSd=${bs} \
        --b1PriorScale=${bs} --SEED=${SEED} --RESULTS_DIR=$RESULTS_DIR \
        --EXP_NAME=$exp_name --MCMC_ITER=${MCMC_ITER} --BURN=${BURN} \
        --b0TunerInit=${betaTunerInit} --b1TunerInit=${betaTunerInit}"

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
