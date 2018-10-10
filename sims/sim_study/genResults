#!/usr/bin/env python3

import sys
import os
import glob
import subprocess


def lsRec(path):
    return list(glob.iglob('{}/**'.format(path), recursive=True))


def dirContains(path, word):
    allFiles = lsRec(path)
    for f in allFiles:
        if word in f:
            return True
    return False


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: ./genResults <path-to-results>")
        sys.exit(1)

    resDir = sys.argv[1] #'results/'
    
    res = list(filter(lambda d: dirContains(resDir+d, 'jld2'), os.listdir(resDir)))

    successes = []
    failures = []

    for d in res:
        out = subprocess.run(['julia', 'post_process.jl', resDir+d])
        if out.returncode == 0:
            print("Successful for {}".format(d))
            successes.append(d) 
        else:
            print("Unsuccessful for {}".format(d))
            failures.append(d) 

    print("successes")
    for s in successes:
        print(s)

    print("failures")
    for f in failures:
        print(f)

