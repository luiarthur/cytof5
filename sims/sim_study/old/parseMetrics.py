#!/usr/bin/env python3
import sys
import os
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
        y = list(filter(lambda x: "{} => ".format(header)in x, lines))[0]
        y = y.split("=>")[1].strip()
        return float(y)

    if len(sys.argv) < 2:
        print("Usage: ./parseMetrics <path-to-results>")
        sys.exit(1)
    else:
        results_dir = sys.argv[1]
        sim_dirs = list(filter(lambda d: "I3_" in d, os.listdir(results_dir)))

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
        os.makedirs('{}/img'.format(results_dir), exist_ok=True)

        LPML = np.array(LPML)
        plt.figure()
        plt.plot(K[order], LPML[order], linestyle='--', marker='o')
        plt.ylabel('LPML')
        plt.xlabel('K')
        plt.savefig('{}/img/lpml.pdf'.format(results_dir), bbox_inches='tight')

        DIC = np.array(DIC)
        plt.figure()
        plt.plot(K[order], DIC[order], linestyle='--', marker='o')
        plt.ylabel('DIC')
        plt.xlabel('K')
        plt.savefig('{}/img/dic.pdf'.format(results_dir), bbox_inches='tight')

        pD = np.array(pD)
        plt.figure()
        plt.plot(K[order], pD[order], linestyle='--', marker='o')
        plt.ylabel('pD')
        plt.xlabel('K')
        plt.savefig('{}/img/pD.pdf'.format(results_dir), bbox_inches='tight')

        Dmean = np.array(Dmean)
        plt.figure()
        plt.plot(K[order], Dmean[order], linestyle='--', marker='o')
        plt.ylabel('Dmean')
        plt.xlabel('K')
        plt.savefig('{}/img/Dmean.pdf'.format(results_dir), bbox_inches='tight')

