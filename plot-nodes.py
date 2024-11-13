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

def load_data_from_csv(path, experiment, round, seed, alpha):
    files = glob.glob(f'{path}experiment-{experiment}*seed-{seed}*alpha-{alpha}_globalRound-{round}.csv')
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

    metrics = {
        "battery": ['reward[min]', 'reward[max]', 'reward[mean]',
                    'batteryPercentage[min]', 'batteryPercentage[max]', 'batteryPercentage[mean]',
                    'localComponentsPercentage[min]', 'localComponentsPercentage[max]', 'localComponentsPercentage[mean]',
                    ],
        "costs": ['reward[min]', 'reward[max]', 'reward[mean]',
                  'localComponentsPercentage[min]', 'localComponentsPercentage[max]', 'localComponentsPercentage[mean]',],
        "aggregated": ['reward[mean]', 'localComponentsPercentage[mean]']
    }

    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    mean_battery = []
    mean_cost = []


    for alpha in alphas:

        for experiment in experiments:
            Path(f'{charts_dir}{experiment}/').mkdir(parents=True, exist_ok=True)


            for seed in range(2,3):

                aggregated_rewards = []
                aggregated_local_components_percentage = []

                for round in range(1, 61):
                    dataframes = load_data_from_csv(data_path, experiment, round, seed, alpha)
                    df = dataframes[0]
                    df = df.dropna()
                    aggregated_rewards.append(df['reward[mean]'].mean())
                    aggregated_local_components_percentage.append(df['localComponentsPercentage[mean]'].mean())
                    if round == 60:
                        battery = df['batteryPercentage[mean]'].mean()
                        cost = df['totalCost[mean]'].mean()
                        mean_battery.append(battery)
                        mean_cost.append(cost)
                    # plot(df, df, round, experiment, metrics[experiment])
                    # df_concat = pd.concat(dataframes).dropna().reset_index().groupby("index")
                    # mean = df_concat.mean()
                    # std = df_concat.std()
                    # aggregated_rewards.append(mean['reward[mean]'].mean())
                    # aggregated_local_components_percentage.append(mean['localComponentsPercentage[mean]'].mean())
                    # plot(mean, std, round, experiment, metrics[experiment])


                # plot_aggregated(aggregated_rewards, f'{experiment}-reward[mean]-seed{seed}', alpha)
                # plot_aggregated(aggregated_local_components_percentage, f'{experiment}-localComponentsPercentage[mean]-seed{seed}', alpha)

    print(mean_battery)

    plot_pareto_battery_costs(mean_battery, mean_cost, alphas)