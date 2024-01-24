import os
from CMFX_source import *
from constants import *
import shutil
import pandas as pd
from itertools import product

def run_sweep(name, variables={}, reset=False):
    if reset:
        results = pd.DataFrame()

        # Create directory for parameter sweeps
        if not os.path.isdir(f'{root}/{sweep_folder}'):
            os.mkdir(f'{root}/{sweep_folder}')

        if not os.path.isdir(f'{root}/{sweep_folder}/{name}'):
            os.mkdir(f'{root}/{sweep_folder}/{name}')
        
        # Remove subdirectories at the beginning of each run
        for subdir in os.listdir(f'{root}/{sweep_folder}/{name}'):
            path = f'{root}/{sweep_folder}/{name}/{subdir}'
            if os.path.isdir(path) and file_prefix in subdir:
                shutil.rmtree(path)

        # Need to unpack the variables and their values into a list of keyword arguments to iterate over
        values_list = list(product(*list(variables.values())))
        kwargs_list = [{variable: value for variable, value in zip(variables.keys(), values)} for values in values_list]
        
        for i, kwargs in enumerate(kwargs_list):
            print(f'Starting run {i + 1}/{len(kwargs_list)}: {kwargs}')
            # Make a new folder for each run
            subdir = f'{file_prefix}_{i+1:03}'
            path = f'{root}/{sweep_folder}/{name}/{subdir}'
            os.mkdir(path)

            # Create and run the simulation
            source = CMFX_Source(**kwargs)
            source.run(path)
            result = source.read_results()
            for variable, value in kwargs.items():
                result[variable] = value
            results = pd.concat([results, result])

        results.to_csv(f'{root}/{sweep_folder}/{name}/results.csv')

    else:
        try:
            results = pd.read_csv(f'{root}/{sweep_folder}/{name}/results.csv')
        except FileNotFoundError as error:
            print(error)
            print('Results file does not yet exist. Please set "reset" to True.')

    return results