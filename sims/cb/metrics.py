#!/usr/bin/env python3
import sys
import os
import subprocess
import numpy as np
import re
import matplotlib.pyplot as plt


def readFile(fname):
    with open(fname, 'r') as f:
        contents = f.readlines()
    return contents


def writeFile(x, fname):
    with open(fname, 'w') as f:
        f.write(x)


if __name__ == '__main__':
    def parse(lines, header):
        y = list(filter(lambda x: "{} => ".format(header) in x, lines))[0]
        y = y.split("=>")[1].strip()
        return float(y)

    if len(sys.argv) < 2:
        print("Usage: python3 metrics <path-to-results>")
        sys.exit(1)
    else:
        results_dir = sys.argv[1]

        sim_dirs = os.listdir(results_dir)
        sim_dirs = list(filter(lambda d: 'K_MCMC' in d, sim_dirs))
        print("Removing K_MCMC = 14")
        sim_dirs = list(filter(lambda d: "K_MCMC14" not in d, sim_dirs))
        DIC = []
        pD = []
        Dmean = []
        LPML = []

        K = np.array([int(re.findall(r'(?<=K_MCMC)\d+', d)[0])
                      for d in sim_dirs])
        order = np.argsort(K)

        # DIC
        for d in sim_dirs:
            c = readFile('{}/{}/log.txt'.format(results_dir, d))
            DIC.append(parse(c, 'DIC'))
            pD.append(parse(c, 'pD'))
            Dmean.append(parse(c, 'Dmean'))
            LPML.append(parse(c, 'LPML'))

        # Plot
        METRICS_DIR = '{}/metrics/'.format(results_dir)
        os.makedirs(METRICS_DIR, exist_ok=True)

        LPML = np.array(LPML)
        plt.figure()
        plt.plot(K[order], LPML[order], linestyle='--', marker='o')
        plt.ylabel('LPML')
        plt.xlabel('K')
        plt.xticks(K)
        plt.savefig('{}/lpml.pdf'.format(METRICS_DIR), bbox_inches='tight')
        plt.close()

        DIC = np.array(DIC)
        plt.figure()
        plt.plot(K[order], DIC[order], linestyle='--', marker='o')
        plt.ylabel('DIC')
        plt.xlabel('K')
        plt.xticks(K)
        plt.savefig('{}/dic.pdf'.format(METRICS_DIR), bbox_inches='tight')
        plt.close()

        pD = np.array(pD)
        plt.figure()
        plt.plot(K[order], pD[order], linestyle='--', marker='o')
        plt.ylabel('pD')
        plt.xlabel('K')
        plt.xticks(K)
        plt.savefig('{}/pD.pdf'.format(METRICS_DIR), bbox_inches='tight')
        plt.close()

        Dmean = np.array(Dmean)
        plt.figure()
        plt.plot(K[order], Dmean[order], linestyle='--', marker='o')
        plt.ylabel('Dmean')
        plt.xlabel('K')
        plt.xticks(K)
        plt.savefig('{}/Dmean.pdf'.format(METRICS_DIR), bbox_inches='tight')
        plt.close()

