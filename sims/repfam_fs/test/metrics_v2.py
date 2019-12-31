import collections
import os
import sys
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rcparams

KTRUE = [4, 5]

def parse_Rs(path_to_Rs_csv):
    Rs = pd.read_csv(path_to_Rs_csv).rename(columns=dict(mean='Mean'))
    return Rs

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


def count_num_small_phenotypes(path, thresh=.01):
    rgx = lambda f: re.match('W\d+_hat', f)
    w_hat_paths = list(filter(rgx, os.listdir(path)))
    num_small_phenotypes = 0
    for wpath in w_hat_paths:
        wi = np.genfromtxt('{}/{}'.format(path, wpath))
        # num_small_phenotypes += ((0 < wi) * (wi < thresh)).sum()
        num_small_phenotypes += (wi < thresh).sum()
    return num_small_phenotypes


def compute_num_selected_features(path):
    rgx = lambda f: re.match('W\d+_hat', f)
    w_hat_paths = list(filter(rgx, os.listdir(path)))
    di = []
    for wpath in sorted(w_hat_paths):
        wi = np.genfromtxt('{}/{}'.format(path, wpath))
        di.append((wi > 0).sum())
    return di


def get_metrics_for_each_dir(results_dir, thresh=.01):
    # Create dict to store results
    out = dict()

    # Traverse results
    for root, dirs, files in os.walk(results_dir):
        for f in files:
            if f == 'log.txt':
                path_to_log = '{}/{}'.format(root, f)

                # Parse LPML / DIC
                metrics = parse_log(path_to_log)

                # Parse W
                path_to_W = '{}/img/yz/txt/'.format(root)
                num_small_phenotypes = count_num_small_phenotypes(path_to_W,
                                                                  thresh)
                metrics['num_selected_features'] = compute_num_selected_features(path_to_W)
                metrics['num_small_phenotypes'] = num_small_phenotypes

                # Parse R
                path_to_R = '{}/img/txt/'.format(root)
                R_df = parse_Rs('{}/Rs.csv'.format(path_to_R))
                metrics['I'] = R_df.shape[0]
                metrics['R_mean'] = R_df.Mean.to_numpy()
                metrics['R_lower'] = R_df.p_02_5.to_numpy()
                metrics['R_upper'] = R_df.p_97_5.to_numpy()
                # metrics['R_mean'] = R_df.p_50_0.to_numpy()
                # metrics['R_lower'] = R_df.p_25_0.to_numpy()
                # metrics['R_upper'] = R_df.p_75_0.to_numpy()

                # Parse Rprob
                path_to_Rprob = path_to_R
                R_prob = np.loadtxt('{}/prob_R_equals_K.txt'.format(path_to_R))
                metrics['Rprob'] = R_prob.T

                # Append to metrics 
                out[path_to_log] = metrics

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
        kmcmc, z, scale, _, seed, _ = path.split('/')
        kmcmc_int = int(kmcmc.replace('KMCMC', ''))
        scale_float = float(scale.replace('scale', ''))
        new_key = (z, scale_float, seed)
        if new_key not in exp_dict:
            exp_dict[new_key] = dict()
        exp_dict[new_key][kmcmc_int] = all_metrics[key]

    return exp_dict


def graph_for_setting(setting, exp_dict, metric, label, labels=None):
    d = exp_dict[setting]
    if metric == 'num_small_phenotypes':
        lpml = []
        num_small = []
        ks = []
        for kmcmc in sorted(d.keys()):
            ks.append(kmcmc)
            lpml.append(d[kmcmc]['LPML'])
            num_small.append(d[kmcmc]['num_small_phenotypes'])
        plt.plot(num_small, lpml, marker='o', label=label)
        plt.xlabel('number of obscure phenotypes')
        plt.ylabel('LPML')
    elif metric == 'Rprob':
        K_min = list(d.keys())[0]
        I = d[K_min]['I']

        if labels is not None:
            if len(labels) == 2:
                c = {labels[0]: 'blue', labels[1]: 'red'}
            else:
                print('NotImplemented!')
            
        for i in range(I):
            plt.subplot(I, 1, i + 1)
            ks = []
            Ri_prob_equals_K_TRUE = []

            Ks = sorted(d.keys())
            for kmcmc in Ks:
                ks.append(kmcmc)
                if kmcmc < KTRUE[i]:
                    Ri_prob_equals_K_TRUE.append(0)
                else:
                    Ri_prob_equals_K_TRUE.append(d[kmcmc]['Rprob'][i, KTRUE[i] - 1])

            plt.plot(ks, Ri_prob_equals_K_TRUE,
                     color=c[label], marker='o', label=label)
            plt.xlabel('KMCMC')
            plt.ylabel('Prob(Ri = K_TRUE)')
            plt.ylim([-0.1, 1.1])
    elif metric == 'R':
        K_min = list(d.keys())[0]
        I = d[K_min]['I']

        if labels is not None:
            if len(labels) == 2:
                c = {labels[0]: 'blue', labels[1]: 'red'}
            else:
                print('NotImplemented!')
            
        for i in range(I):
            plt.subplot(I, 1, i + 1)
            ks = []
            Ri_mean = []
            Ri_lower = []
            Ri_upper = []

            Ks = sorted(d.keys())
            for kmcmc in Ks:
                ks.append(kmcmc)
                Ri_mean.append(d[kmcmc]['R_mean'][i])
                Ri_lower.append(d[kmcmc]['R_lower'][i])
                Ri_upper.append(d[kmcmc]['R_upper'][i])

            plt.plot(ks, Ri_mean, color=c[label], marker='o', label=label)
            plt.fill_between(ks, Ri_lower, Ri_upper,
                             color=c[label], alpha=.3)
            plt.xlabel('KMCMC')
            plt.ylabel('R_{}'.format(i + 1))
            plt.yticks(range(min(Ks) - 2, int(max(Ri_upper) + .5), 2),
                       range(min(Ks) - 2, int(max(Ri_upper) + .5), 2))
    else:
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
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        results_dir = 'results/test-sims-5-5'

    print('Results dir: {}'.format(results_dir))

    # Get a dictionary indexed by experiment setting (z, scale, seed)
    exp_dict = get_exp_dict(results_dir) 
    
    # Metrics to plot
    # metrics = ['LPML', 'DIC', 'num_small_phenotypes', 'R']
    # metrics = ['LPML', 'DIC', 'num_small_phenotypes']
    metrics = ['LPML', 'DIC', 'R']

    # Name of metrics dir
    metrics_dir = '{}/metrics'.format(results_dir) 

    # Get unique zs
    zs = set([key[0] for key in exp_dict.keys()])
    print('zs: {}'.format(zs))

    # Get unique seeds
    seeds = set([key[2] for key in exp_dict.keys()])
    print('seeds: {}'.format(seeds))

    # Get unique scales
    scales = set([key[1] for key in exp_dict.keys()])
    num_scales = len(scales)
    print('scales: {}'.format(scales))


    # sorted exp_dict keys
    exp_dict_keys_sorted = sorted(exp_dict.keys())

    # TODO:
    # graph Rs
    labels = ['scale={}'.format(scale)for scale in scales]

    for z in zs:
        for seed in seeds:
            for metric in metrics:
                for setting in exp_dict_keys_sorted:
                    zidx, scale, sd = setting
                    if z == zidx and sd == seed:
                        label = 'scale={}'.format(scale)
                        graph_for_setting(setting, exp_dict, metric, label,
                                          labels=labels)
                dest_dir = '{}/{}/{}'.format(metrics_dir, z, seed)
                if metric == 'R':
                    plt.legend(loc='lower right')
                elif metric == 'Rprob':
                    plt.legend(loc='lower center')
                else:
                    plt.legend()
                plt.tight_layout()
                # Make destination dir if needed
                os.makedirs(dest_dir, exist_ok=True)
                plt.savefig('{}/{}.pdf'.format(dest_dir, metric))
                plt.close()
