from itertools import cycle
import os
import sys
import re

import boto3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
# mpl.use('Agg')  # Non-interactive plot 
mpl.use("TkAgg")  # Interactive plot

import Timer


# Image directory
img_dir = '/scratchdata/alui2/cytof/results/cb_best/img/tsne/init_neg6'

# Path to TSNE output
path_to_tsne = '{}/cb_paper_Y_with_tsne.csv'.format(img_dir)

# Read in data
Y = pd.read_csv(path_to_tsne)

# Markers for plotting
# markers_orig = list('ov^<>spP*hHXDd1234')
markers_orig = list('ov^<>spP*hHXD')

# Number of samples
I = Y.sample_id.drop_duplicates().count()

# Only include markers if w_{ik} > w_thresh for at least one sample 
w_thresh = .05
# w_thresh = .1

# W estimate
N = Y.groupby('sample_id').lam_est.count()
W = (Y.groupby(['sample_id', 'lam_est']).lam_est.count() / N).unstack()
W = W.fillna(0)

# Subpopulations that have at least w_thresh presence in
# at least one sample.
lam_are_prominent = (W > w_thresh).any(0)
prominent_lam = lam_are_prominent[lam_are_prominent].index

# Markers to be used.
colors_orig = ['C{}'.format(i) for i in range(10)] + ['red']
colors_orig = cycle(colors_orig)
colors = dict((lam, c) for lam, c in zip(prominent_lam, colors_orig))
markers = dict((lam, m) for lam, m in zip(prominent_lam, markers_orig))
    
# Set random seed
seed = 1
np.random.seed(seed)


# TODO: Black and white
for i in range(I):
    Ni = (Y.sample_id == i).sum()
    N_thresh = 100000
    if Ni > N_thresh:
        groups = Y[Y.sample_id == i].sample(N_thresh).groupby('lam_est')
    else:
        groups = Y[Y.sample_id == i].groupby('lam_est')
    fig, ax = plt.subplots()
    num_clus = len(groups)
    for name, group in groups:
        # wi = len(group) / min(N_thresh, Ni)
        lami = group.lam_est.iloc[0]
        wik = W[lami][i]
        # if lami in prominent_lam:
        if wik > w_thresh:
            ms = 3
            label = '{}: {:.2f}%'.format(lami, wik * 100)
            marker = markers[lami]
            color = colors[lami]
        else:
            ms = 0
            label = None
            marker = None
            color = None
        ax.plot(group.tsne_embedding_0, group.tsne_embedding_1,
                # marker='o', linestyle='', ms=10 * wik)
                # marker=marker, linestyle='', ms=10 * wik, color='black')
                marker=marker, linestyle='', ms=ms,
                label=label, alpha=.5, color=color)

    # Set alpha=1 to legend markers
    leg = ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), mode='expand',
                    loc=3, ncol=4, handletextpad=0.1)

    for lh in leg.legendHandles:
        lh._legmarker.set_alpha(1)
        lh._legmarker.set_ms(10)

    # plt.legend(bbox_to_anchor=(1, 1))

    # plt.tight_layout()
    # plt.show()
    plt.savefig('{}/tsne_sample_{}.pdf'.format(img_dir, i + 1),
                bbox_inches='tight')
    plt.close()

