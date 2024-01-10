import os
from CMFX_source import *
import shutil
import pandas as pd
from multiprocessing import Pool

reset = True

gas_pressure = 4560 # Torr
# GEOMETRY [all length units in cm]
distances = np.linspace(50, 70, 1)
He3_diameter = 0.96 * IN2CM
He3_length = 8 * IN2CM

detector_diameter = 1 * IN2CM
detector_length = 11.34 * IN2CM
detector_offset = 0.87 * IN2CM

HDPE_height = detector_length
HDPE_diameter = 2 * IN2CM

void_radius = 100

plasma_outerRadius = 25
plasma_innerRadius = 5
plasma_length = 60

particles = 50000

folder = '/Users/Nick/programs/openmc/CMFX-Neutronics/CMFX'
file_prefix = 'run'

results = pd.DataFrame()

def plot_tallies(result):
    energy_bins_high_edge = sorted(result['energy high [eV]'].unique())
    energy_bins_low_edge = sorted(result['energy low [eV]'].unique())

    for high_energy_edge, low_energy_edge in zip(energy_bins_high_edge, energy_bins_low_edge):
        filtered_df =  result[result['energy high [eV]']==high_energy_edge]
        plt.loglog(
            filtered_df["time low [s]"],
            filtered_df["mean"],
            label=f'{low_energy_edge:.0e}eV to {high_energy_edge:.0e}eV'
        )

    plt.legend()
    plt.ylim(bottom=1e-11)
    plt.xlabel('Time [s]')
    plt.ylabel('Neutron absorption')
    plt.show()

# Remove subdirectories at the beginning of each run
if reset:
    for subdir in os.listdir(folder):
        path = os.path.join(folder, subdir)
        if os.path.isdir(path) and file_prefix in subdir:
            shutil.rmtree(path)
            
    # with Pool(processes=4) as pool:
    for i, distance in enumerate(distances):
        source = CMFX_Source(gas_pressure, distance, He3_diameter, He3_length, detector_diameter,
                             detector_length, detector_offset, HDPE_diameter, HDPE_height, void_radius,
                             plasma_outerRadius, plasma_innerRadius, plasma_length,
                             particles=particles, create_mesh_tally=False)

        # Make a new folder for each run
        subdir = f'{file_prefix}_{i+1:03}'
        path = os.path.join(folder, subdir)
        os.mkdir(path)

        source.run(path)
        result = source.read_results()
        result['Distance'] = distance
        results = pd.concat([results, result])

        source.plot_flux(section='parallel')

    results.to_csv('results.csv')

else:
    results = pd.read_csv('results.csv')

counts = np.zeros(len(distances))
errors = np.zeros(len(distances))

for i, (distance, grp) in enumerate(results.groupby('Distance')):
    count = sum(grp['mean'])
    std = sum(grp['std. dev.'])
    counts[i] = count * particles
    errors[i] = std * particles

plt.errorbar(distances, counts, yerr=errors)
plt.xlabel('Distance (cm)')
plt.ylabel('Count')
plt.show()