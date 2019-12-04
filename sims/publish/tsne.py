from itertools import cycle
import os
# Limit the number of Openblas threads used.
# Equivalent to `export OMP_NUM_THREADS=4` in bash.
# Needs to be done before importing any library that uses OpenBLAS.
os.environ['OMP_NUM_THREADS'] = '4'
import sys
import re

import boto3
from sklearn.manifold import TSNE
from sklearn.preprocessing import scale
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')  # Non-interactive plot 
# mpl.use("TkAgg")  # Interactive plot

import Timer


if __name__ == '__main__':
    if len(sys.argv) > 1:
        path_to_best_cb_results = sys.argv[1]
        SEED = int(sys.argv[2])
        tsne_aws_bucket = sys.argv[3]
        tsne_aws_destination = sys.argv[4]
    else:
        path_to_best_cb_results = '/scratchdata/alui2/cytof/results/cb_best/img/'
        SEED = 1
        tsne_aws_bucket = 'cytof-fam-paper'
        tsne_aws_destination = '/img/post/cb/best/img'


    # Make results image dir if needed
    img_dir = '{}/tsne'.format(path_to_best_cb_results)
    os.makedirs(img_dir, exist_ok=True)

    # Path to CB Data (transformed expressions)
    path_to_cb_data = '../cb/data/cb.csv'

    # Read Data
    data = pd.read_csv(path_to_cb_data)
    I = data.sample_id.drop_duplicates().count()

    # Replace NaN's with -3 (like init in FAM).
    Y = data.fillna(-6)

    # Only keep rows where all expressions are at least -6. 
    Y = Y[(Y >= -6).all(1)]

    # Get best lambda
    Y['lam_est'] = -1
    for f in os.listdir(path_to_best_cb_results):
        if re.match('lam\d+_best\.txt', f):
            # Read lam_est_i
            lam_est_i = np.loadtxt('/'.join([path_to_best_cb_results, f]))
            # Get sample id
            sample_id = int(re.findall('\d+', f)[0]) - 1
            # Add best lambda estimate to Y
            Y.loc[Y.sample_id == sample_id, 'lam_est'] = lam_est_i.astype(int)

    # Remove markers that are highly missing/negative or positive.
    # These selections were made in the paper.
    good_markers = [True, False, True, False, True, False, True, True, False,
                    True, False, True, True, True, False, True, True, False, False,
                    True, False, True, True, True, True, False, True, False, True,
                    True, False, True]
    is_good_marker = np.where(good_markers)[0]
    good_markers_names = list(Y.columns[is_good_marker])

    # Add embeddings to Y
    for t in range(2):
        Y['tsne_embedding_{}'.format(t)] = np.nan

    # Center and scale data for markers used in analysis.
    # NOTE: Is scaling necessary?
    # Y_scaled = scale(Y[good_markers_names])

    for i in range(I):
        # Create TSNE fitter. Then time and fit the tsne.
        tsne = TSNE(verbose=2, random_state=SEED)
        with Timer.Timer(ndigits=2):
            Yi = Y[Y.sample_id == i][good_markers_names].to_numpy()
            # Yi = Y[Y.sample_id == i][good_markers_names].to_numpy() * 10
            # Yi = (Y[Y.sample_id == i][good_markers_names].to_numpy() > 0) * 1
            tsne.fit(Yi)
            for z in range(2):
                Y.loc[Y.sample_id == i,
                      'tsne_embedding_{}'.format(z)] = tsne.embedding_[:, z]
        print('Sample {}:'.format(i))
        print('KL Divergence: {}'.format(tsne.kl_divergence_))
        print('num iters: {}/{}'.format(tsne.n_iter_ + 1, tsne.n_iter))

    # Write this new df to csv
    Y.to_csv('{}/cb_paper_Y_with_tsne.csv'.format(img_dir), index=False)

    # Markers for plotting
    markers_orig = list('ov^<>spP*hHXDd1234')

    # TODO: Black and white
    for i in range(I):
        markers = cycle(markers_orig)
        groups = Y[Y.sample_id == i].groupby('lam_est')
        fig, ax = plt.subplots()
        num_clus = len(groups)
        Ni = (Y.sample_id == i).sum()
        for name, group in groups:
            wi = len(group) / Ni
            lami = group.lam_est.head(1).item()
            marker = next(markers)
            if wi > .05:
                ms = 4
                label = '{}: {:.2f}%'.format(lami, wi * 100)
            else:
                ms = 1
                label = None
            ax.plot(group.tsne_embedding_0, group.tsne_embedding_1,
                    # marker='o', linestyle='', ms=10 * wi)
                    # marker=marker, linestyle='', ms=10 * wi, color='black')
                    marker=marker, linestyle='', ms=ms,
                    label=label, alpha=.7)
        plt.legend(bbox_to_anchor=(1, 1))
        plt.savefig('{}/tsne_sample_{}.pdf'.format(img_dir, i + 1),
                    bbox_inches='tight')
        plt.close()

# 
# # TSNE for all
# groups = Y.groupby('lam_est')
# fig, ax = plt.subplots()
# num_clus = len(groups)
# for (name, group), marker in zip(groups, list(markers[:num_clus])):
#       ax.plot(group.tsne_embedding_0, group.tsne_embedding_1,
#               marker='o', linestyle='', ms=1)
#               # marker=marker, linestyle='', ms=6, color='black')
# 
# plt.savefig('{}/tsne_sample_all.pdf'.format(img_dir), bbox_inches='tight')
# plt.close()
# 
# # TODO: Try tsne for just single samples.
# 
# # TODO: Send results
