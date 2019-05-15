import os
import sys
import re
import numpy as np
import matplotlib.pyplot as plt

class Result():
    def __init__(self, path):
        self.K_true = int(re.findall('(?<=K)\d+', path)[0])
        self.bs = int(re.findall('(?<=BS)\d+', path)[0])
        self.K_vb = int(re.findall('(?<=K_VB)\d+', path)[0])
        self.path = path

    def get_metric(self, metric):
        with open(self.path, 'r') as f:
            lines = f.readlines()
            rgx = '(?<={}:)[^|]*(?=>|)'.format(metric)
            m = re.findall(rgx, lines[-2])[0]
            return float(m)

if __name__ == '__main__':
    # Get path to results
    if len(sys.argv) > 1:
        path_to_results = sys.argv[1]
    else:
        path_to_results = 'results/vb-sim-paper/'
    
    # search for log files
    results = []
    for root, dirs, files in os.walk(path_to_results):
        if 'log.txt' in files and 'test' not in root:
            d = '{}/log.txt'.format(root)
            results.append(Result(d))

    # Get stuff
    os.makedirs('{}/tuning/'.format(path_to_results), exist_ok=True)
    for K in [5, 10]:
        K_true = list(filter(lambda r: r.K_true == K, results))
        bs = sorted(set(r.bs for r in results))
        kvb = sorted(set(r.K_vb for r in results))

        M = np.zeros((len(bs), len(kvb)))
        for i in range(len(bs)):
            for j in range(len(kvb)):
                res = list(filter(lambda r: r.bs == bs[i] and r.K_vb == kvb[j], K_true))
                res = [r.get_metric('elbo') for r in res]
                if len(res) > 0:
                    M[i, j] = np.mean(res)
                    # M[i, j] = np.max(res)
                else:
                    M[i, j] = float('nan')

        plt.figure()
        cm = plt.get_cmap('Reds')
        cm.set_bad(color='black')
        plt.imshow(M, cmap=cm)
        plt.xlabel('K_VB')
        plt.ylabel('batchsize')
        plt.xticks(range(len(kvb)), kvb)
        plt.yticks(range(len(bs)), bs)
        plt.colorbar()
        plt.savefig('{}/tuning/K{}.pdf'.format(path_to_results, K))
        plt.close()
