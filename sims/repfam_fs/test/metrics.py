import collections
import os
import matplotlib.pyplot as plt
import numpy as np

# Plot settings
fontsize = 15
plt.rcParams['font.size'] = fontsize
plt.rcParams['xtick.labelsize'] = fontsize
plt.rcParams['ytick.labelsize'] = fontsize
plt.rcParams['figure.figsize'] = (6, 5)

# For each Z in (Z1, Z2) 
#     For each scale in (0, .01, .1, 1, 10)
#         Draw a curve for each k in {2, 3, 4, 5}

# zs = ['z1', 'z2']
# scales = [0, 0.01, 0.1, 1, 10]
# ks = [2, 3, 4, 5]

def get_ks(results_dir):
    # Get dir names for each K
    kdirs = sorted(os.listdir(results_dir))
    # Get all values of K
    ks = [d.replace('KMCMC', '') for d in kdirs]

    return ks, kdirs


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


def get_metrics_for_each_dir(results_dir):
    # Create dict to store results
    out = dict()

    # Traverse results
    for root, dirs, files in os.walk(results_dir):
        for f in files:
            if f == 'log.txt':
                path_to_log = '{}/{}'.format(root, f)
                metrics = parse_log(path_to_log)
                out[path_to_log] = metrics
                # TODO:
                # Include number of small phenotypes in metrics

    return out


def get_exp_dict(results_dir):
    # Get Ks and KMCMC dirname
    ks, kdirs = get_ks(results_dir)

    # For each directory, 
    all_metrics = get_metrics_for_each_dir(results_dir)

    # Experiments dictionary, indexed by (z, scale)
    exp_dict = dict()

    # Split all the keys
    for key in all_metrics:
        path = key.replace(results_dir + '/', '')
        kmcmc, z, scale, _ = path.split('/')
        kmcmc_int = int(kmcmc.replace('KMCMC', ''))
        scale_float = float(scale.replace('scale', ''))
        new_key = (z, scale_float)
        if new_key not in exp_dict:
            exp_dict[new_key] = dict()
        exp_dict[new_key][kmcmc_int] = all_metrics[key]

    return exp_dict


def graph_for_setting(setting, exp_dict, metric, label):
    d = exp_dict[setting]
    ks = []
    ms = []

    for kmcmc in sorted(d.keys()):
        ks.append(kmcmc)
        ms.append(d[kmcmc][metric])

    plt.plot(ks, ms, marker='o', label=label)
    plt.xlabel('K')
    plt.ylabel(metric)
    plt.xticks(ks)


if __name__ == '__main__':
    results_dir = 'results/test-sims'
    print('Results dir: {}'.format(results_dir))

    # Get a dictionary indexed by experiment setting (z, scale)
    exp_dict = get_exp_dict(results_dir) 
    
    # Metrics to plot
    metrics = ['LPML', 'DIC']

    # Name of metrics dir
    metrics_dir = '{}/metrics'.format(results_dir) 

    # Get unique zs
    zs = set([key[0] for key in exp_dict.keys()])

    # sorted exp_dict keys
    exp_dict_keys_sorted = sorted(exp_dict.keys())

    for z in zs:
        for metric in metrics:
            for setting in exp_dict_keys_sorted:
                zidx, scale = setting
                if z == zidx:
                    label = 'scale={}'.format(scale)
                    graph_for_setting(setting, exp_dict, metric, label)
            dest_dir = '{}/{}'.format(metrics_dir, z)
            plt.legend()
            plt.tight_layout()
            # Make destination dir if needed
            os.makedirs(dest_dir, exist_ok=True)
            plt.savefig('{}/{}.pdf'.format(dest_dir, metric))
            plt.close()
