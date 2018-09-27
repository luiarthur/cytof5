#!/bin/bash

DEST=$1
aws s3 sync result $DEST

# All buckets:
#DEST="s3://cytof-results/sim/complexZ_thin10y/"
