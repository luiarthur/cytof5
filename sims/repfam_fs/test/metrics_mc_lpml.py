# NOTE: Create a data frame of logs, indexed by directory.

import collections
import os
import sys
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rcparams


def parse_log(path_to_log):
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


def parse_dir(path, log_name='log.txt'):
    metrics_df = pd.DataFrame()

    for root, dirs, files in os.walk(path):
        if log_name in files:
            path_to_log = '{}/{}'.format(root, log_name)
            metrics = parse_log(path_to_log)
            settings = root.replace(path, '').split('/')
            settings = [d for d in settings if d > '']
            for s in settings:
                idx = re.findall('\d+', s)[0]
                key = s.replace(idx, '')
                metrics[key] = int(idx)
            metrics_df = metrics_df.append(
                pd.DataFrame(metrics, index=[0]),
                ignore_index=True,
                sort=False
            )

    return metrics_df


### TEST ###
metrics_df = parse_dir('results/test-sims-5-12')


unique_scales = metrics_df.scale.unique()
num_scales = unique_scales.shape[0]

for i in range(num_scales):
    scale = unique_scales[i]
    plt.subplot(num_scales, 1, i + 1)
    LPML = (metrics_df[metrics_df.scale == scale][['KMCMC', 'LPML']]
            .groupby('KMCMC'))

    LPML_mean = LPML.mean()
    LPML_lower = LPML.quantile(.025)
    LPML_upper = LPML.quantile(.975)

    plt.plot(LPML_upper, marker='o', label='97.5%')
    plt.plot(LPML_mean, marker='o', label='mean')
    plt.plot(LPML_lower, marker='o', label='2.5%')
    plt.title('scale: {}'.format(scale))

plt.legend(loc='lower right')
plt.tight_layout()
plt.show()
