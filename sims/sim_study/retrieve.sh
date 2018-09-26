#!/bin/bash

AWS_BUCKET=$1
aws s3 sync $AWS_BUCKET result/

# For example,
# AWS_BUCKET="s3://cytof-results/Wed-Sep-19-20:46:41-PDT-2018/"
# AWS_BUCKET="s3://cytof-results/sim/complexZ_thin10y/"
