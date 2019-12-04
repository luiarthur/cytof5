import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_all_rprob_files(path_to_sim):
    R_df = pd.DataFrame()
    for root, dirs, files in os.walk(path_to_sim):
        for f in files:
            if f == 'prob_R_equals_K.txt':
                R_prob = np.loadtxt('{}/{}'.format(root, f))
                Kmax, I = R_prob.shape
                seed = int(re.findall('(?<=seed)\d+', root)[0])
                scale = int(re.findall('(?<=scale)\d+', root)[0])
                kmcmc = int(re.findall('(?<=KMCMC)\d+', root)[0])
                R_df_mini = pd.DataFrame()
                for i in range(I):
                    R_df_mini_i = pd.DataFrame({
                        'prob': R_prob[:, i],
                        'k': np.arange(Kmax) + 1,
                        'Kmax': Kmax,
                        'kmcmc': kmcmc,
                        'seed': seed,
                        'scale': scale,
                        'i': i + 1
                    })

                    R_df_mini = R_df_mini.append(R_df_mini_i,
                                                 ignore_index=True,
                                                 sort=False)

                R_df = R_df.append(R_df_mini, ignore_index=True, sort=False)
                
    return R_df


if __name__ == '__main__':
    if len(sys.argv) > 1: 
        results_dir = sys.argv[1]
    else:
        results_dir = 'results/test-sims-5-5/'

    print('pid: {}'.format(os.getpid()))

    # NOTE: This is the true K (modify accordingly)
    K_TRUE = [4, 5]

    # NOTE: This number of samples (modify accordingly)
    I = len(K_TRUE)

    # Make dataframe from results
    R_df = get_all_rprob_files(results_dir)

    # Get unique settings
    seeds = sorted(R_df.seed.drop_duplicates().to_numpy())
    scales = sorted(R_df.scale.drop_duplicates().to_numpy())
    kmcmcs = sorted(R_df.kmcmc.drop_duplicates().to_numpy())
    samples = sorted(R_df.i.drop_duplicates().to_numpy())

    # Plot results
    for seed in seeds:
        fig, ax = plt.subplots(I, 1, sharex=False)
        for i in range(I):
            (R_df[(R_df.seed == seed) & (R_df.i == i + 1) & 
                  (R_df.k == K_TRUE[i])]
             .sort_values('kmcmc')
             .pivot(index='kmcmc', values='prob', columns='scale')
             .plot(marker='o', ax=ax[i]))
            ax[i].set_ylim(-.1, 1.1)
            ax[i].set_xlim(R_df.kmcmc.min() - 1, R_df.kmcmc.max() + 1)
            ax[i].set_title('seed: {}, i: {}'.format(seed, i + 1))
            ax[i].set_ylabel('Pr(R{} = R{}_TRUE)'.format(i + 1, i + 1))

        # Save figure
        dest_dir = '{}/metrics/z3/seed{}'.format(results_dir, seed)
        os.makedirs(dest_dir, exist_ok=True)
        plt.tight_layout()
        plt.savefig('{}/prob_R{}_equals_KTRUE.pdf'.format(dest_dir, i + 1))
        plt.close()

