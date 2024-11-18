import os
import re
import glob
import matplotlib
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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


def load_csv(path, experiment, global_round, alpha, beta, gamma, seed):
    files = glob.glob(f'{path}/experiment-{experiment}*seed-{seed}*_alpha-{alpha}_beta-{beta}_gamma-{gamma}_globalRound-{global_round}_*.csv')
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


def load_csv_density(path, experiment, global_round, seed):
    files = glob.glob(f'{path}/experiment-density-nodes-{experiment}-augmented_seed-{seed}*_globalRound-{global_round}_randomAugmentedSeed-*.csv')
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
        return 'Local Components (\%)'
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
        plt.tight_layout()
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
        plt.plot(time, mean, color='#440154')
        plt.fill_between(time, lower, upper, color='#440154', alpha=0.2)
        label = beautify_label(metric)
        plt.title(f'{label} - $\\alpha$ = {alpha}, $\\beta$ = {beta}, $\gamma$ = {gamma}')
        plt.xlabel('Global Round')
        plt.ylabel(label)
        plt.tight_layout()
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
        colors = {c: viridis(c) for c in data[color].loc[i]}
        plt.scatter(data[all_nodes_x].loc[i], data[all_nodes_y].loc[i], color=colormapping)
        plt.title(f'Time {i}')
        legend_elements = [mpatches.Patch(color=c, label=f"{p * 100}%") for p, c in sorted(colors.items())]
        plt.ylabel("Distance (dam)")
        plt.xlabel("Distance (dam)")
        plt.legend(handles=legend_elements, title="Local Components (\%)", loc = 'upper left')
        plt.tight_layout()
        plt.savefig(f'{path}/time-{i}.pdf')
        plt.close()

def isValid(alpha, beta, gamma):
    s = alpha + beta + gamma
    return s >= 0.99 and s <= 1.01


if __name__ == '__main__':

    # Set matplotlib parameters
    matplotlib.rcParams.update({'axes.titlesize': 18})
    matplotlib.rcParams.update({'axes.labelsize': 18})
    matplotlib.rcParams.update({'xtick.labelsize': 15})
    matplotlib.rcParams.update({'ytick.labelsize': 15})
    matplotlib.rcParams.update({"text.usetex": True})
    matplotlib.rc('text.latex', preamble=r'\usepackage{amsmath,amssymb,amsfonts}')

    # Experiments parameters
    data_path   = 'data'
    charts_path = 'charts'
    experiment  = 'mixed'
    min_seed    = 0
    max_seed    = 3
    step_seed   = 1
    rounds      = 60
    alphas      = { 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 }
    betas       = { 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 }
    gammas      = { 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 }
    metrics     = ['reward[mean]', 'localComponentsPercentage[mean]', 'batteryPercentage[mean]', 'totalCost[mean]', 'componentsInCloud[mean]', 'componentsInInfrastructural[mean]', 'loss[mean]']

    Path(charts_path).mkdir(parents=True, exist_ok=True)


    cartesian_product = list(product(alphas, betas, gammas))
    print(len(cartesian_product))
    cartesian_product = [(a, b, g) for a, b, g in cartesian_product if isValid(a, b, g)]
    print(len(cartesian_product))

    for alpha, beta, gamma in cartesian_product:
    # alpha, beta, gamma = 0.7, 0.3, 0.0
        aggregated = { m: [] for m in metrics }
        for global_round in range(1, rounds + 1):
            data = load_csv(data_path, experiment, global_round, alpha, beta, gamma, 0)
            found = len(data)
            if found > max_seed + 1:
                raise Exception(f'Too many files, please check the regex! [DEBUG] alpha {alpha} beta {beta} gamma {gamma}')
            elif found == 0:
                raise Exception(
                    f'No files found, please check the regex! [DEBUG] alpha {alpha} beta {beta} gamma {gamma}')
            if  found > 0:
                print(f'Charting alpha {alpha}, beta {beta}, gamma {gamma}')
                print(f'Found {len(data)} files')
                data_concat = pd.concat(data).dropna().reset_index().groupby('index')
                mean = data_concat.mean()
                std = data_concat.std()
                for metric in metrics:
                    # aggregated[metric].append((mean[metric].mean(), std[metric].mean()))
                    d = data[0]
                    aggregated[metric].append((d[metric].mean(), 0.0))
                # plot(mean, std, global_round, metrics, alpha, beta, gamma, charts_path)
        if len(aggregated[metrics[0]]) > 0:
            plot_aggregated(aggregated, metrics, rounds, alpha, beta, gamma, charts_path)


    # Charting experiment with density (AC augmented)

    # max_seed    = 5
    # rounds      = 120
    # charts_path = 'charts/density'
    #
    # Path(charts_path).mkdir(parents=True, exist_ok=True)
    #
    # for seed in range(max_seed):
    #     for global_round in range(1, rounds + 1, 10):
    #         data = load_csv_density('data-density', 'AC', global_round, seed)
    #         plot_density_scatter(data, seed, global_round, charts_path)
    #
    # # Charting experiment with density (No AC augmented)
    #
    # rounds = 120
    # charts_path = 'charts/density/NO-AC'
    #
    # Path(charts_path).mkdir(parents=True, exist_ok=True)
    #
    # for global_round in range(1, rounds + 1, 10):
    #     data = load_csv_density('data-density', 'NO-AC' ,global_round, 0)
    #     plot_density_scatter(data, 0, global_round, charts_path)
