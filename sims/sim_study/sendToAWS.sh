#!/bin/bash

DEST="s3://cytof-sim-results/"
aws s3 sync results $DEST

