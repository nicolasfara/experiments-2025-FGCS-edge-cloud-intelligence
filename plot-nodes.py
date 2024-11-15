import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
import glob
import re


def extractVariableNames(filename):
    """
    Gets the variable names from the Alchemist data files header.

    Parameters
    ----------
    filename : str
        path to the target file

    Returns
    -------
    list of list
        A matrix with the values of the csv file

    """
    with open(filename, 'r') as file:
        dataBegin = re.compile(r'\d')
        lastHeaderLine = ''
        for line in file:
            if dataBegin.match(line[0]):
                break
            else:
                lastHeaderLine = line
        if lastHeaderLine:
            regex = re.compile(r' (?P<varName>\S+)')
            return regex.findall(lastHeaderLine)
        return []

def openCsv(path):
    """
    Converts an Alchemist export file into a list of lists representing the matrix of values.

    Parameters
    ----------
    path : str
        path to the target file

    Returns
    -------
    list of list
        A matrix with the values of the csv file

    """
    regex = re.compile(r'\d')
    with open(path, 'r') as file:
        lines = filter(lambda x: regex.match(x[0]), file.readlines())
        return [[float(x) for x in line.split()] for line in lines]

def load_data_from_csv(path, experiment, round, seed):
    files = glob.glob(f'{path}experiment-{experiment}-nodes_seed-{seed}*_globalRound-{round}.csv')
    dataframes = []
    print(f'For round {round} loaded {len(files)} files')
    for file in files:
        print(file)
        columns = extractVariableNames(file)
        data = openCsv(file)
        df = pd.DataFrame(data, columns=columns)
        dataframes.append(df)
    return dataframes

def plot(mean, std, round, experiment, metrics):
    for metric in metrics:
        plt.plot(mean['time'], mean[metric], color='#440154')
        # plt.fill_between(mean['time'], mean[metric] - std[metric], mean[metric] + std[metric], color='#440154', alpha=0.2)
        plt.title(f'Global Round {round}')
        plt.xlabel('Time')
        plt.ylabel(metric)
        plt.savefig(f'charts/{experiment}/globalRound-{round}_{metric}.pdf')
        plt.close()

def plot_aggregated(mean, metric, alpha):
    time = np.linspace(1, len(mean), len(mean))
    plt.plot(time, mean, color='#440154')
    plt.title(f'Aggregated {metric} alpha {alpha}')
    plt.xlabel('Global Round')
    plt.ylabel(metric)
    plt.savefig(f'charts/aggregated_{metric}-alpha{alpha}.pdf')
    plt.close()


def plot_pareto_battery_costs(mean_battery, mean_costs, alphas):

    colormap = plt.cm.get_cmap('viridis', len(alphas))
    colors = [colormap(i) for i in range(len(alphas))]
    plt.plot(mean_costs, mean_battery, linestyle='-', color='black', linewidth=1, zorder=1)
    plt.scatter(mean_costs, mean_battery, s=40, color=colors, zorder=2)
    plt.title(f'Pareto')
    plt.xlabel('Costs')
    plt.ylabel('Battery')
    # plt.yscale('function', functions=(np.arcsinh, np.sinh))
    plt.savefig(f'charts/pareto.pdf')
    plt.close()


if __name__ == '__main__':
    # experiments = ['battery', 'costs']
    experiments = ['mixed']
    charts_dir = 'charts/'
    Path(charts_dir).mkdir(parents=True, exist_ok=True)
    data_path = 'data/'
    data = load_data_from_csv(data_path, 'density', 40, 0)[0]
    all_nodes = [f"node-{i}" for i in range(1, 48)]
    all_nodes_x = [f"{node}-x" for node in all_nodes]
    all_nodes_y = [f"{node}-y" for node in all_nodes]
    color = [f"{node}[localComponentsPercentage]" for node in all_nodes]
    viridis = plt.cm.get_cmap('viridis', 3)
    print(color)
    for i in range(0, 100, 5):
        colormapping = [viridis(c) for c in data[color].loc[i]]
        plt.scatter(data[all_nodes_x].loc[i], data[all_nodes_y].loc[i], color=colormapping)
        plt.title(i)
        plt.savefig(f"{i}.png")
        plt.close()
