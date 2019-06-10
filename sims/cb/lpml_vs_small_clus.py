#!/usr/bin/env python3
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import re

def num_small_clusters(ws, thresh=0.01):
    num_small_clus = 0
    for wi in ws:
        for wik in wi:
            num_small_clus += wik <= thresh
    return num_small_clus
        

def get_lpml(path):
    with open(path, 'r') as f:
        contents = f.readlines()
    lpml_line = list(filter(lambda line: 'LPML =>' in line, contents))[0]
    lpml = re.findall('(?<=LPML\s=>\s)-\d+\.\d+', lpml_line)[0]
    return float(lpml)

def read_ws(path_to_imgdir):
    files = os.listdir(path_to_imgdir)
    w_files = list(filter(lambda f: re.match('W_\d+_hat\.txt', f) is not None, files))
    ws = []
    for wf in w_files:
        with open('{}/{}'.format(path_to_imgdir, wf), 'r') as f:
            wi = f.readlines()
            wi = [float(wik) for wik in wi]
            ws.append(wi)
    return ws


if __name__ == '__main__':
    path_to_results = 'results/cb-paper/'

    # threshold to classify as small cluster
    THRESH = .01

    # model dict: K => (number of W_{ik} < 1%, lpml)
    model = {}

    for root, dirs, files in os.walk(path_to_results):
        if 'K_MCMC' in root and 'log.txt' in files:
            path = '{}/{}'.format(root, 'log.txt')
            K = int(re.findall('(?<=K_MCMC)\d+', path)[0])
            lpml = get_lpml(path)
            ws = read_ws('{}/img'.format(root))
            num_small_clus = num_small_clusters(ws, thresh=THRESH)
            model[K] = (num_small_clus, lpml)

    # print(model)

    # 3 columns: K, num small clusters, lpml
    results = np.zeros((len(model), 3))

    i = 0
    for K in sorted(model):
        results[i, 0] = K
        results[i, 1] = model[K][0]
        results[i, 2] = model[K][1]
        i += 1

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(results[:, 1],  results[:, 2])
    ax.scatter(results[:, 1],  results[:, 2])
    for K in model:
        ax.annotate(K, model[K], size=12)
    plt.scatter(model[21][0], model[21][1], marker='X', s=100)
    plt.show()
