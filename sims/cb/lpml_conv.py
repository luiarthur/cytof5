import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import rcparams

def parse_log(path_to_log, metric='lpml'):
    with open(path_to_log, 'r') as f:
        contents = f.read()
    lpml = re.findall('(?<=LPML:\s)-*\d+.\d+', contents)
    kmcmc = re.findall('(?<=K_MCMC)\d+', path_to_log)[0]
    return int(kmcmc), np.array(lpml, dtype=float)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        results_dir = 'results/cb-paper'


    # Results dict
    results_dict = dict()

    # Traverse results
    for root, dirs, files in os.walk(results_dir):
        for f in files:
            if f == 'log.txt':
                path_to_log = '{}/{}'.format(root, f)
                kmcmc, lpml = parse_log(path_to_log)
                results_dict[kmcmc] = lpml

    # Get unique seeds and kmcmc
    kmcmcs = sorted(set(results_dict.keys()))

    # Plot LPML
    for kmcmc in kmcmcs:
        plt.plot(results_dict[kmcmc], lw=2, label=kmcmc)
    plt.legend(title='KMCMC')
    plt.ylabel('LPML')
    plt.xlabel('Every 10th iteration in MCMC after burn-in')
    plt.tight_layout()
    plt.show()

