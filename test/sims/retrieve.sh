#!/bin/bash

SERVER=$1
DIR="repo/Cytof5/test/sims/result/"

rsync -av $SERVER:~/$DIR/ result/
