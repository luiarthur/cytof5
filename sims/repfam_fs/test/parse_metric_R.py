import itertools
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import rcparams

from parse_log_metrics import logpath_to_dict

def parse_Rs(path_to_Rs_csv):
    Rs = pd.read_csv(path_to_Rs_csv).rename(columns=dict(mean='Mean'))
    Rs['sample_id'] = range(Rs.shape[0])
    return Rs


def crawl_dirs_R(results_dir, logfname):
    # Create dataframe to store results
    df = pd.DataFrame()

    # Crawl results_dir
    for root, dirs, files in os.walk(results_dir):
        for f in files:
            if f == logfname:
                path_to_log = '{}/{}'.format(root, f)

                # Parse R
                path_to_Rs_csv = '{}/img/txt/Rs.csv'.format(root)
                R_df = parse_Rs(path_to_Rs_csv)

                # log dictionay
                log_dict = logpath_to_dict(results_dir, path_to_log)

                # append to result
                for key in log_dict:
                    R_df[key] = log_dict[key]

                df = df.append(R_df, ignore_index=True)

    return df

if __name__ == '__main__':
    if sys.argv[0] == '':
        # NOTE: For testing
        results_dir = '/scratchdata/arthur/cytof/results/repfam/test-sims-6-2'
        logfname = 'log-out.txt'
    elif len(sys.argv) >= 2:
        results_dir = sys.argv[1]
        if len(sys.argv) >= 3:
            logfname = sys.argv[2]
        else:
            logfname = 'log-out.txt'
    else:
        print('Usage: python3 {} <path-to-results-dir>'.format(sys.argv[0]))
        sys.exit()

    # Make directories to store results in necessary.
    metrics_dir = '{}/metrics'.format(results_dir)
    os.makedirs(metrics_dir, exist_ok=True)

    print('results dir: {}'.format(results_dir))
    print('log file name: {}'.format(logfname))

    Rs_df = crawl_dirs_R(results_dir, logfname)
    Rs_df.to_csv('{}/Rs.csv'.format(metrics_dir), index=False)

    # NOTE: Custom stuff. test-sims >= 6.2 
    # for seed in sorted(Rs.seed.unique()):
    #     for metric in ['DIC', 'LPML']:

    unique_sample_ids = sorted(Rs_df.sample_id.unique())
    num_samples = len(unique_sample_ids)

    for seed in sorted(Rs_df.seed.unique()):
        plt.figure()
        for sample_id in unique_sample_ids:
            df = Rs_df[(Rs_df.seed == seed) & (Rs_df.sample_id == sample_id)]
            df = df.sort_values('Kmcmc')
            df = df.pivot(index='Kmcmc',  columns='scale', values='Mean')
            ax = plt.subplot(num_samples, 1, sample_id + 1)
            df.plot(marker='o', ax=ax)
            # TODO: UQ?
            Kmcmcs = df[0].keys().astype(int)
            Rmax = df.to_numpy().max()
            Rmin = df.to_numpy().min()
            plt.xticks(Kmcmcs)
            plt.xlim([Kmcmcs.min() - 1, Kmcmcs.max() + 1])
            plt.ylim([Rmin - .5, Rmax + .5])
            plt.ylabel('R_{}'.format(sample_id + 1))
            plt.legend(loc='lower right', title='scale')
            if sample_id == 0:
                plt.title('seed: {}'.format(seed))
            if sample_id == unique_sample_ids[-1]:
                plt.xlabel('Kmcmc')
        # NOTE: save images
        # plt.show()
        out_dir = '{}/seed_{}'.format(metrics_dir, int(seed))
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig('{}/R.pdf'.format(out_dir), bbox_inches='tight')
        plt.close()

