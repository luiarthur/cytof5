import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re

def parse_log(path_to_log, metric='lpml'):
    with open(path_to_log, 'r') as f:
        contents = f.read()
    lpml = re.findall('(?<=LPML:\s)-*\d+.\d+', contents)
    seed = re.findall('(?<=seed)\d+', path_to_log)[0]
    scale = re.findall('(?<=scale)\d+', path_to_log)[0]
    kmcmc = re.findall('(?<=KMCMC)\d+', path_to_log)[0]
    return int(seed), int(kmcmc), float(scale), np.array(lpml, dtype=float)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        results_dir = 'results/test-sims-5-7'


    # Results dict
    results_dict = dict()

    # Traverse results
    for root, dirs, files in os.walk(results_dir):
        for f in files:
            if f == 'log.txt':
                path_to_log = '{}/{}'.format(root, f)
                seed, kmcmc, scale, lpml = parse_log(path_to_log)
                results_dict[seed, kmcmc, scale] = lpml

    # Get unique seeds and kmcmc
    seeds = sorted(set([key[0] for key in results_dict]))
    kmcmcs = sorted(set([key[1] for key in results_dict]))
    scales = sorted(set([key[2] for key in results_dict]))

    # Plot LPML
    # for seed in seeds:
    for seed in [seeds[1]]:
        for scale in [scales[1]]:
            for kmcmc in kmcmcs:
                plt.plot(results_dict[seed, kmcmc, scale], lw=2, label=kmcmc)
            plt.legend(title='KMCMC')
            plt.show()

    # plt.plot(kmcmcs, [results_dict[seeds[2], kmcmc, scales[0]][-1]
    #                   for kmcmc in kmcmcs])
    # plt.show()
