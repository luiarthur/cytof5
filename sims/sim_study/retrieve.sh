#!/bin/bash

AWS_BUCKET="s3://cytof-sim-results"
aws s3 sync $AWS_BUCKET results/

