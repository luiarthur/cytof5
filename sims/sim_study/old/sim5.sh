#!/bin/bash

# Maximum number of cores to use
MAX_CORES=7

# AWS Bucket to store results
AWS_BUCKET="s3://cytof-vary-kmcmc-n10000"

# STAGGER_TIME in seconds. To avoid mass dumping to disk simultaneously. 
STAGGER_TIME=100

# Experiment settings
MCMC_ITER=1000
BURN=10000
I=3
J=32
N_factor="10000"
K=8
L=4
K_MCMC="6 7 8 9 10 11 12"
L_MCMC=5
betaPriorScale="0.3162"
betaTunerInit="0.1"
RESULTS_DIR="results/sim5/"
SEED="98 64"
fix_b1="true"
printFreq=50

# simulation number, just for book keeping. Ignore this.
simNumber=0

# Testing
if [[ $@ == **--test** ]]
then
  echo "Testing with small numbers..."
  STAGGER_TIME=0

  # Experiment settings
  MCMC_ITER=10
  BURN=10
  K_MCMC=10
  L_MCMC=5
  betaPriorScales="0.01"
  betaTunerInit="0.01"
  SEED="98"
  printFreq=2
fi


for seed in $SEED; do
  for nFac in $N_factor; do
    for bs in $betaPriorScale; do
      for k_mcmc in $K_MCMC; do
        # Simulation number
        simNumber=$((simNumber + 1)) 

        # Experiment name
        exp_name="I${I}_J${J}_N_factor${nFac}_K${K}_L${L}_K_MCMC${k_mcmc}_L_MCMC${L_MCMC}_betaPriorScale${bs}_betaTunerInit${betaTunerInit}_fix_b1_SEED${seed}"

        # Output directory
        outdir="$RESULTS_DIR/$exp_name/"
        mkdir -p $outdir

        # Julia Command to run
        jlCmd="julia sim.jl --I=${I} --J=${J} --N_factor=${nFac} --K=${K} \
          --L=${L} --K_MCMC=${k_mcmc} --L_MCMC=${L_MCMC} --b0PriorSd=${bs} \
          --b1PriorScale=${bs} --SEED=${seed} --RESULTS_DIR=$RESULTS_DIR \
          --EXP_NAME=$exp_name --MCMC_ITER=${MCMC_ITER} --BURN=${BURN} \
          --b0TunerInit=${betaTunerInit} --b1TunerInit=${betaTunerInit} \
          --fix_b1=${fix_b1} --printFreq=${printFreq}"

        # Sync results to S3
        syncToS3="aws s3 sync $outdir $AWS_BUCKET/$exp_name --exclude '*.nfs*'"

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
done
