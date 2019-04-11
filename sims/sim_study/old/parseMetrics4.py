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
        y = list(filter(lambda x: "{} => ".format(header)in x, lines))[0]
        y = y.split("=>")[1].strip()
        return float(y)

    if len(sys.argv) < 2:
        print("Usage: ./parseMetrics <path-to-results>")
        sys.exit(1)
    else:
        results_dir = sys.argv[1]

        def parse_sim(n:int, scale:float, init:float, seed:int):
            sim_dirs = subprocess.run('echo {}/*N_factor{}_*Scale{}*Init{}_*SEED{}*'.format(
                results_dir, n, scale, init, seed), shell=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout
            sim_dirs = [ d.decode('utf-8').split('/')[-1] for d in sim_dirs.split() ]
            for d in sim_dirs:
                print(d)
            print()

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
            METRICS_DIR = '{}/metrics/N_factor{}_betaPriorScale{}_betaTunerInit{}_SEED{}'
            METRICS_DIR = METRICS_DIR.format(results_dir, n, scale, init, seed)
            os.makedirs(METRICS_DIR, exist_ok=True)

            LPML = np.array(LPML)
            plt.figure()
            plt.plot(K[order], LPML[order], linestyle='--', marker='o')
            plt.ylabel('LPML')
            plt.xlabel('K')
            plt.savefig('{}/lpml.pdf'.format(METRICS_DIR), bbox_inches='tight')
            plt.close()

            DIC = np.array(DIC)
            plt.figure()
            plt.plot(K[order], DIC[order], linestyle='--', marker='o')
            plt.ylabel('DIC')
            plt.xlabel('K')
            plt.savefig('{}/dic.pdf'.format(METRICS_DIR), bbox_inches='tight')
            plt.close()

            pD = np.array(pD)
            plt.figure()
            plt.plot(K[order], pD[order], linestyle='--', marker='o')
            plt.ylabel('pD')
            plt.xlabel('K')
            plt.savefig('{}/pD.pdf'.format(METRICS_DIR), bbox_inches='tight')
            plt.close()

            Dmean = np.array(Dmean)
            plt.figure()
            plt.plot(K[order], Dmean[order], linestyle='--', marker='o')
            plt.ylabel('Dmean')
            plt.xlabel('K')
            plt.savefig('{}/Dmean.pdf'.format(METRICS_DIR), bbox_inches='tight')
            plt.close()

        #n = 1000
        #scale = 0.1
        #init = 0.1
        #seed = 98
        #parse_sim(n, scale, init, seed)

        sim_dirs = list(filter(lambda d: "I3_" in d, os.listdir(results_dir)))
        N = set([int(n) for n in re.findall(r'(?<=N_factor)\d+', ','.join(sim_dirs))])
        SCALE = set([float(s) for s in re.findall(r'(?<=Scale)\d+\.\d+', ','.join(sim_dirs))])
        INIT = set([float(i) for i in re.findall(r'(?<=Init)\d+\.\d+', ','.join(sim_dirs))])
        SEED = set([int(s) for s in re.findall(r'(?<=SEED)\d+', ','.join(sim_dirs))])
        for n in N:
            for scale in SCALE:
                for init in INIT:
                    for seed in SEED:
                        parse_sim(n, scale, init, seed)
