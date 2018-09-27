#!/bin/bash

if [ "$#" -ne 1 ]; then
  echo "usage: . retrieve.sh <bucket/path>"
  echo "Here are the buckets available in s3://cytof-results/"
  aws s3 ls s3://cytof-results/ --recursive
else
  AWS_BUCKET=$1
  aws s3 sync $AWS_BUCKET result/
fi

# All buckets:
# AWS_BUCKET="s3://cytof-results/sim/complexZ_thin10y/"
