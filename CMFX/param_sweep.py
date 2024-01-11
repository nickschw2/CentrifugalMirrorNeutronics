import os
from CMFX_source import *
from constants import *
import shutil
import pandas as pd

def run_sweep(name, variable, values, reset=False, plot=False):
    results = pd.DataFrame()
    if reset:
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
                
        # with Pool(processes=4) as pool:
        for i, value in enumerate(values):
            source = CMFX_Source(**{variable: value})

            # Make a new folder for each run
            subdir = f'{file_prefix}_{i+1:03}'
            path = f'{root}/{sweep_folder}/{name}/{subdir}'
            os.mkdir(path)

            source.run(path)
            result = source.read_results()
            result[variable] = value
            results = pd.concat([results, result])

        results.to_csv(f'{root}/{sweep_folder}/{name}/results.csv')

    else:
        results = pd.read_csv(f'{root}/{sweep_folder}/{name}/results.csv')

    counts = np.zeros(len(values))
    errors = np.zeros(len(values))

    for i, (val, grp) in enumerate(results.groupby(variable)):
        count = sum(grp['mean'])
        std = sum(grp['std. dev.'])
        counts[i] = count * particles
        errors[i] = std * particles

    if plot:
        plt.errorbar(values, counts, yerr=errors)
        plt.xlabel(f'{variable}')
        plt.ylabel('Count')
        plt.show()

    return counts, errors