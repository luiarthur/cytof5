#!/bin/bash

PATH_TO_RESULTS='results/test-sims-5-11/'

PATH_TO_LOG='z3/scale10/seed1'

# Get mus0_1
mus01=`cat $PATH_TO_RESULTS/KMCMC{{05..07},15}/$PATH_TO_LOG/img/txt/mus0_mean.txt \
  | grep -oP '^-\d+\.\d+'`


lpml=`cat $PATH_TO_RESULTS/KMCMC{{05..07},15}/$PATH_TO_LOG/log.txt \
  | grep -oP '(?<=LPML\s=>\s)-\d+\.\d+'`

echo "${mus01[@]/%/$'\n'}" > /tmp/mus01.txt
echo "${lpml[@]/%/$'\n'}" > /tmp/lpml.txt
R -q -e 'mus01 = read.table("/tmp/mus01.txt")[[1]]; lpml = read.table("/tmp/lpml.txt")[[1]]; plot(mus01, lpml)'

evince Rplots.pdf
rm Rplots.pdf
