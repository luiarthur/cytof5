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
        if len(sys.argv) > 2:
            excludes = sys.argv[2]
        else:
            excludes = "NOTHING"
        print("excludes: {}".format(excludes))

        sim_dirs = os.listdir(results_dir)
        sim_dirs = list(filter(lambda d: 'K_MCMC' in d, sim_dirs))
        sim_dirs = list(filter(lambda d: not re.search(excludes, d), sim_dirs))

        L = np.array([int(re.findall(r'(?<=L0_MCMC)\d+', d)[0])
                      for d in sim_dirs])
        L = set(L)
        print('L: {}'.format(L))

        for l in L:
            DIC = []
            pD = []
            Dmean = []
            LPML = []
            
            sim_dirs_l = filter(lambda d: 'L0_MCMC{}'.format(l) in d, sim_dirs)
            sim_dirs_l = sorted(sim_dirs_l)
        
            K = np.array([int(re.findall(r'(?<=K_MCMC)\d+', d)[0])
                          for d in sim_dirs_l])
            order = np.argsort(K)

            # DIC
            for d in sim_dirs_l:
                c = readFile('{}/{}/log.txt'.format(results_dir, d))
                DIC.append(parse(c, 'DIC'))
                pD.append(parse(c, 'pD'))
                Dmean.append(parse(c, 'Dmean'))
                LPML.append(parse(c, 'LPML'))

            # Plot
            METRICS_DIR = '{}/metrics/L0_MCMC{}/'.format(results_dir, l)
            os.makedirs(METRICS_DIR, exist_ok=True)

            label_fs = 20
            x_fs = 24
            y_fs = 24
            plt.rc('font', size=20)
            ms=12

            LPML = np.array(LPML)
            plt.figure()
            plt.plot(K[order], LPML[order], linestyle='--', marker='o', markersize=ms)
            plt.ylabel('LPML', fontsize=label_fs)
            plt.xlabel('K', fontsize=label_fs)
            plt.xticks(K, rotation=90, fontsize=x_fs)
            plt.yticks(fontsize=y_fs)
            plt.savefig('{}/lpml.pdf'.format(METRICS_DIR), bbox_inches='tight')
            plt.close()

            DIC = np.array(DIC)
            plt.figure()
            plt.plot(K[order], DIC[order], linestyle='--', marker='o', markersize=ms)
            plt.ylabel('DIC', fontsize=label_fs)
            plt.xlabel('K', fontsize=label_fs)
            plt.xticks(K, rotation=90, fontsize=x_fs)
            plt.yticks(fontsize=y_fs)
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            plt.savefig('{}/dic.pdf'.format(METRICS_DIR), bbox_inches='tight')
            plt.close()

            pD = np.array(pD)
            plt.figure()
            plt.plot(K[order], pD[order], linestyle='--', marker='o', markersize=ms)
            plt.ylabel('pD', fontsize=label_fs)
            plt.xlabel('K', fontsize=label_fs)
            plt.xticks(K, rotation=90, fontsize=x_fs)
            plt.yticks(fontsize=y_fs)
            plt.savefig('{}/pD.pdf'.format(METRICS_DIR), bbox_inches='tight')
            plt.close()

            Dmean = np.array(Dmean)
            plt.figure()
            plt.plot(K[order], Dmean[order], linestyle='--', marker='o', markersize=ms)
            plt.ylabel('Dmean', fontsize=label_fs)
            plt.xlabel('K', fontsize=label_fs)
            plt.xticks(K, rotation=90, fontsize=x_fs)
            plt.yticks(fontsize=y_fs)
            plt.savefig('{}/Dmean.pdf'.format(METRICS_DIR), bbox_inches='tight')
            plt.close()

