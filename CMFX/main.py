from CMFX_source import *
from constants import *
from param_sweep import *

import sys
sys.path.append('../')
from plot_style import *

### STANDARD SIMULATION ###
path = f'{root}/{examples_folder}/standard'
reset = True
if reset:
    if os.path.isdir(path):
        shutil.rmtree(path)
    source = CMFX_Source(create_mesh_tally=True)
    source.run(path)
    results = source.read_results()
    results.to_csv(f'{path}/results.csv')
else:
    try:
        result = pd.read_csv(f'{path}/results.csv')
        with open(f'{path}/{source_pkl}', 'rb') as file:
            source = pickle.load(file)
    except FileNotFoundError as error:
        print(error)
        print('Results file does not yet exist. Please set "reset" to True.')

source.plot_flux(section='perp')
source.plot_flux(section='parallel')

### CONVERGENCE SWEEP ###
N_particles = np.logspace(3, 6, 16).astype('int')
name = 'convergence'
variables = {'particles': N_particles}
convergenceResults = run_sweep(name, variables, reset=True)
convergenceResults['MCSE'] = convergenceResults['std. dev.'] / convergenceResults['mean']

convergenceResults.plot(x='particles', y='MCSE', logx=True)
plt.xlabel('# Particles')
plt.ylabel('Monte Carlo Error')
plt.show()

### DETECTOR LOCATION SWEEP ###
radialDistances = np.linspace(50, 90, 17)
axialDistances = np.linspace(0, 30, 5)
name = 'detector_location'
variables = {'radialDistance': radialDistances, 'axialDistance': axialDistances}
detectorLocationResults = run_sweep(name, variables, reset=True)

fig, ax = plt.subplots()
for axialDistance, grp in detectorLocationResults.groupby('axialDistance'):
    grp.plot(ax=ax, x='radialDistance', y='mean', label=f'z={int(axialDistance)} (cm)')
    ax.fill_between(grp['radialDistance'], grp['mean'] - grp['std. dev.'], grp['mean'] + grp['std. dev.'], alpha=0.2)

ax.set_xlabel('Radial Distance (cm)')
ax.set_ylabel('Counts')
ax.legend()
plt.show()