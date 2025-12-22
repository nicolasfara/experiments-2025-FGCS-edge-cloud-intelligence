import os
import re
import glob
import matplotlib
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product
import matplotlib.pyplot as plt

def extract_variable_names(filename):
    with open(filename, 'r') as file:
        data_begin = re.compile(r'\d')
        last_header_line = ''
        for line in file:
            if data_begin.match(line[0]):
                break
            else:
                last_header_line = line
        if last_header_line:
            regex = re.compile(r' (?P<varName>\S+)')
            return regex.findall(last_header_line)
        return []


def open_csv(path):
    regex = re.compile(r'\d')
    with open(path, 'r') as file:
        lines = filter(lambda x: regex.match(x[0]), file.readlines())
        return [[float(x) for x in line.split()] for line in lines]


def load_csv(path, experiment, global_round, alpha, beta, gamma):
    files = glob.glob(f'{path}/experiment-{experiment}*_alpha-{alpha}*_beta-{beta}*_gamma-{gamma}*_globalRound-{global_round}_*.csv')
    found = len(files)
    dataframes = []
    if found > 0:
        #print(f'Found {found} files for alpha {alpha} beta {beta} gamma {gamma}')
        for file in files:
            columns = extract_variable_names(file)
            data = open_csv(file)
            df = pd.DataFrame(data, columns=columns)
            dataframes.append(df)
    return dataframes


def load_csv_density(path, global_round, seed):
    files = glob.glob(f'{path}/experiment-density-nodes-AC-augmented_seed-{seed}*_globalRound-{global_round}_randomAugmentedSeed-*.csv')
    print(f'GR {global_round} seed {seed} found {len(files)} files')
    for file in files:
        columns = extract_variable_names(file)
        data = open_csv(file)
        df = pd.DataFrame(data, columns=columns)
        return df


def beautify_label(label):
    if 'reward[mean]' in label:
        return 'Reward'
    elif 'localComponentsPercentage[mean]' in label:
        return 'Local Components \%)'
    elif 'batteryPercentage[mean]' in label:
        return 'Battery (\%)'
    elif 'totalCost[mean]' in label:
        return 'Total Cost'
    elif 'componentsInCloud[mean]' in label:
        return 'Components In Cloud (\%)'
    elif 'componentsInInfrastructural[mean]' in label:
        return 'Components In Edge Server (\%)'
    elif 'loss[mean]' in label:
        return 'Loss'
    else:
        raise Exception('Unknown Label')

def plot(mean, std, global_round, metrics, alpha, beta, gamma, output_path):

    path = f'{output_path}/alpha-{alpha}_beta-{beta}_gamma-{gamma}'
    Path(path).mkdir(parents=True, exist_ok=True)

    for metric in metrics:
        plt.plot(mean['time'], mean[metric], color='#440154')
        plt.fill_between(mean['time'], mean[metric] - std[metric], mean[metric] + std[metric], color='#440154', alpha=0.2)
        plt.title(f'Global Round {global_round} - $\\alpha$ = {alpha}, $\\beta$ = {beta}, $\gamma$ = {gamma}')
        plt.xlabel('Time')
        plt.ylabel(beautify_label(metric))
        plt.savefig(f'{path}/globalRound-{global_round}_{metric}.pdf')
        plt.close()

def extract_mean_std(data):
    mean = [m for m, _ in data]
    std  = [s for _, s in data]
    return mean, std

def plot_aggregated(aggregated, metrics, rounds, alpha, beta, gamma, output_path):

    path = f'{output_path}/aggregated/alpha-{alpha}_beta-{beta}_gamma-{gamma}'
    Path(path).mkdir(parents=True, exist_ok=True)

    for metric in metrics:
        mean, std = extract_mean_std(aggregated[metric])
        time = np.linspace(1, rounds, len(mean))
        upper = [m + s for m, s in zip(mean, std)]
        lower = [m - s for m, s in zip(mean, std)]
        color = '#365c8d'

        local = '#c2df23'
        edge = '#1e9b8a'
        cloud = '#fde725'

        plt.plot(time, mean, color=color)
        plt.fill_between(time, lower, upper, color=color, alpha=0.2)
        label = beautify_label(metric)
        if alpha == 0.99:
            alpha = 1.0
        elif beta == 0.99:
            beta = 1.0
        elif gamma == 0.99:
            gamma = 1.0

        lw = 2
        
        if 'Battery' in label:
            plt.axhline(y=100, color=edge, linestyle='--', linewidth=lw, label='Edge/Cloud')
            plt.axhline(y=91, color=local, linestyle='--', linewidth=lw, label='Local')
        elif 'Cost' in label:
            plt.axhline(y=0, color=local, linestyle='--', linewidth=lw, label='Local')
            plt.axhline(y=160, color=cloud, linestyle='--', linewidth=lw, label='Cloud')
            plt.axhline(y=25, color=edge, linestyle='--', linewidth=lw, label='Edge')

        plt.title(f'{label} - $\\alpha$ = {alpha}, $\\beta$ = {beta}, $\gamma$ = {gamma}')
        plt.xlabel('Global Round')
        plt.ylabel(label)
        if metric in ['batteryPercentage[mean]', 'totalCost[mean]']:
            plt.legend(loc = 'center right')
        plt.savefig(f'{path}/{metric}.pdf')
        plt.close()


def plot_density_scatter(data, seed, global_round, output_path):

    path = f'{output_path}/seed-{seed}/round-{global_round}'
    Path(path).mkdir(parents=True, exist_ok=True)
    max_time = 100
    all_nodes = [f"node-{i}" for i in range(48)]
    all_nodes_x = [f"{node}-x" for node in all_nodes]
    all_nodes_y = [f"{node}-y" for node in all_nodes]
    color = [f"{node}[localComponentsPercentage]" for node in all_nodes]
    viridis = plt.cm.get_cmap('viridis', 3)
    for i in range(max_time):
        colormapping = [viridis(c) for c in data[color].loc[i]]
        plt.scatter(data[all_nodes_x].loc[i], data[all_nodes_y].loc[i], color=colormapping)
        plt.title(f'Time {i}')
        plt.savefig(f'{path}/time-{i}.pdf')
        plt.close()


def check_params(alpha, beta, gamma):
    return (alpha == 1.0 and beta == 0.0 and gamma == 0.0) or (alpha == 0.0 and beta == 1.0 and gamma == 0.0) or (alpha == 0.0 and beta == 0.0 and gamma == 1.0)
    # return True
if __name__ == '__main__':

    # Set matplotlib parameters
    matplotlib.rcParams.update({'axes.titlesize': 18})
    matplotlib.rcParams.update({'axes.labelsize': 18})
    matplotlib.rcParams.update({'xtick.labelsize': 15})
    matplotlib.rcParams.update({'ytick.labelsize': 15})
    matplotlib.rcParams.update({"text.usetex": True})
    matplotlib.rcParams.update({'legend.fontsize': 20})
    matplotlib.rcParams.update({'legend.title_fontsize': 20})
    matplotlib.rc('text.latex', preamble=r'\usepackage{amsmath,amssymb,amsfonts}')

    # Experiments parameters
    data_path   = 'data-learning-mixed'
    charts_path = 'charts-fgcs'
    experiment  = 'mixed'
    min_seed    = 0
    max_seed    = 3
    step_seed   = 1
    rounds      = 60
    alphas      = { 0.0, 1.0 }
    betas       = { 0.0, 1.0 }
    gammas      = { 0.0, 1.0 }
    metrics     = ['reward[mean]', 'localComponentsPercentage[mean]', 'batteryPercentage[mean]', 'totalCost[mean]', 'componentsInCloud[mean]', 'componentsInInfrastructural[mean]', 'loss[mean]']

    Path(charts_path).mkdir(parents=True, exist_ok=True)


    cartesian_product = list(product(alphas, betas, gammas))
    
    for alpha, beta, gamma in cartesian_product:
        aggregated = {m: [] for m in metrics}
        for global_round in range(1, rounds + 1):
            data = load_csv(data_path, experiment, global_round, alpha, beta, gamma)
            if len(data) > 0:
                data_concat = pd.concat(data).dropna().reset_index().groupby('index')
                mean = data_concat.mean()
                std = data_concat.std()
                for metric in metrics:
                    aggregated[metric].append((mean[metric].mean(), std[metric].mean()))
                # plot(mean, std, global_round, metrics, alpha, beta, gamma, charts_path)
        if len(aggregated[metrics[0]]) > 0 and check_params(alpha, beta, gamma):
            plot_aggregated(aggregated, metrics, rounds, alpha, beta, gamma, charts_path)


    # Charting experiment with density

    # max_seed    = 5
    # rounds      = 120
    # charts_path = 'charts/density'

    # Path(charts_path).mkdir(parents=True, exist_ok=True)

    # for seed in range(max_seed):
    #     for global_round in range(1, rounds + 1, 10):
    #         data = load_csv_density('data-density', global_round, seed)
    #         plot_density_scatter(data, seed, global_round, charts_path)