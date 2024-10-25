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

def load_data_from_csv(path, experiment, round):
    files = glob.glob(f'{path}experiment-{experiment}*globalRound-{round}.csv')
    dataframes = []
    print(f'For round {round} loaded {len(files)} files')
    for file in files:
        columns = extractVariableNames(file)
        data = openCsv(file)
        df = pd.DataFrame(data, columns=columns)
        dataframes.append(df)
    return dataframes

def plot(mean, std, round, experiment, metrics):
    for metric in metrics:
        plt.plot(mean['time'], mean[metric], color='#440154')
        plt.fill_between(mean['time'], mean[metric] - std[metric], mean[metric] + std[metric], color='#440154', alpha=0.2)
        plt.title(f'Global Round {round}')
        plt.xlabel('Time')
        plt.ylabel(metric)
        plt.savefig(f'charts/{experiment}/globalRound-{round}_{metric}.pdf')
        plt.close()

if __name__ == '__main__':
    experiments = ['battery', 'costs']
    charts_dir = 'charts/'
    Path(charts_dir).mkdir(parents=True, exist_ok=True)
    data_path = 'data/'

    metrics = {
        "battery": ['reward[min]', 'reward[max]', 'reward[mean]',
                    'batteryPercentage[min]', 'batteryPercentage[max]', 'batteryPercentage[mean]',
                    'localComponentsPercentage[min]', 'localComponentsPercentage[max]', 'localComponentsPercentage[mean]',
                    ],
        "costs": ['reward[min]', 'reward[max]', 'reward[mean]',
                    'localComponentsPercentage[min]', 'localComponentsPercentage[max]', 'localComponentsPercentage[mean]',]
    } 

    for experiment in experiments: 
        Path(f'{charts_dir}{experiment}/').mkdir(parents=True, exist_ok=True)

        for round in range(1, 16):
            if round > 10 and "battery" in experiment:
                break
            dataframes = load_data_from_csv(data_path, experiment, round)
            df_concat = pd.concat(dataframes)
            by_row_index = df_concat.groupby(df_concat.index)
            mean = by_row_index.mean()
            mean = mean.dropna()
            var = by_row_index.var()
            var = var.dropna()
            plot(mean, var, round, experiment, metrics[experiment])