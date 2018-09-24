#!/bin/bash

I=3
J=32
NFactor="100 1000 10000"
K=4
L=4
K_MCMC=10
L_MCMC=5

for n in $NFactor
do
  echo "Experiment with NFactor = $n"
  julia sim.jl $I $J $n $K $L $K_MCMC $L_MCMC &
done
