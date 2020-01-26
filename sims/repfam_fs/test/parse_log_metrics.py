import sys
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rcparams

def parse_log_metrics(path_to_log):
    # Read log file
    with open(path_to_log, "r") as f:
        contents = f.read()

    # Keep only the metrics
    metrics = contents.split('metrics:')[1]
    # Separate into lines
    metrics = metrics.split('\n')
    # keep only lines with '=>'
    metrics = list(filter(lambda line: '=>' in line, metrics))

    # Create empty dict to return
    out = dict()

    # store metrics in a dict
    for metric in metrics:
        key, val = metric.split('=>')
        out[key.strip()] = float(val.strip())
    
    return out


def split_logpath(results_dir, path_to_log):
    colnames = path_to_log.replace(results_dir, '')
    colnames = colnames.split('/')[1:-1]
    return colnames


def logpath_to_dict(results_dir, path_to_log):
    colnames = split_logpath(results_dir, path_to_log)
    d = dict()

    for c in colnames:
        key, value = c.split('_')
        d[key] = float(value)
    
    return d


def crawl_dirs_metrics(results_dir, logfname):
    # Create dataframe to store results
    df = pd.DataFrame()

    # Crawl results_dir
    for root, dirs, files in os.walk(results_dir):
        for f in files:
            if f == logfname:
                path_to_log = '{}/{}'.format(root, f)

                # Parse metrics
                metrics = parse_log_metrics(path_to_log)

                # log dictionay
                log_dict = logpath_to_dict(results_dir, path_to_log)

                # append to result
                record = {**metrics, **log_dict}
                df = df.append(record, ignore_index=True)

    return df




if __name__ == '__main__':
    # if len(sys.argv) >= 2:
    #     results_dir = sys.argv[1]
    #     if len(sys.argv) >= 3:
    #         logfname = sys.argv[2]
    #     else:
    #         logfname = 'log-out.txt'
    # else:
    #     print('Usage: python3 {} <path-to-results-dir>'.format(sys.argv[0]))
    #     sys.exit()
    results_dir = '/scratchdata/arthur/cytof/results/repfam/test-sims-6-2'
    logfname = 'log-out.txt'

    print('results dir: {}'.format(results_dir))
    print('log file name: {}'.format(logfname))

    metrics_df = crawl_dirs_metrics(results_dir, logfname)
    print(metrics_df)

# NOTE: Custom stuff
for seed in sorted(metrics_df.seed.unique()):
    for metric in ['DIC', 'LPML']:
        df = metrics_df[(metrics_df.seed == seed)]
        df = df.sort_values('Kmcmc')
        df = df.pivot(index='Kmcmc',  columns='scale', values=metric)
        df.plot(marker='o')
        Kmcmcs = df[0].keys().astype(int)
        plt.xticks(Kmcmcs)
        plt.xlim([Kmcmcs.min() - 1, Kmcmcs.max() + 1])
        plt.xlabel('Kmcmc')
        plt.ylabel(metric)
        plt.title('seed: {}'.format(seed))
        plt.tight_layout()
        plt.show()

