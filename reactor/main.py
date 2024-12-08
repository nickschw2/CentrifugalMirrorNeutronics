from reactor_source import *
from constants import *
from param_sweep import *

import sys
sys.path.append('../')
from plot_style import *

### STANDARD SIMULATION ###
path = f'{root}/{examples_folder}/standard'
reset = False
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

source.plot_flux()

### CONVERGENCE SWEEP ###
N_particles = np.logspace(3, 6, 16).astype('int')
name = 'convergence'
variables = {'particles': N_particles}
convergenceResults = run_sweep(name, variables=variables, reset=False)
convergenceResults['MCSE'] = convergenceResults['std. dev.'] / convergenceResults['mean']

convergenceResults.plot(x='particles', y='MCSE', logx=True)
plt.xlabel('# Particles')
plt.ylabel('Monte Carlo Error')
plt.savefig(f'{figures_folder}/neutronics_convergence.png', dpi=600)
plt.show()

### DETECTOR LOCATION SWEEP ###
radialDistances = np.linspace(50, 90, 17)
axialDistances = np.linspace(0, 30, 5)
name = 'detector_location'
variables = {'radialDistance': radialDistances, 'axialDistance': axialDistances}
detectorLocationResults = run_sweep(name, variables=variables, reset=False)

fig, ax = plt.subplots()
for axialDistance, grp in detectorLocationResults.groupby('axialDistance'):
    grp.plot(ax=ax, x='radialDistance', y='mean', label=f'z={int(axialDistance)} (cm)')
    ax.fill_between(grp['radialDistance'], grp['mean'] - grp['std. dev.'], grp['mean'] + grp['std. dev.'], alpha=0.2)

ax.set_xlabel('Radial Distance (cm)')
ax.set_ylabel('Counts/s')
ax.legend()
ax.set_ylim(bottom=0)
plt.savefig(f'{figures_folder}/neutronics_location_sweep.png', dpi=600)
plt.show()

### TEMPERATURE AND DENSITY SWEEP ###
Ti_values = np.logspace(np.log10(0.3), 1, 15) # keV
ni_values = np.logspace(12, 14, 5) # cm^-3
name = 'plasma_properties'
variables = {'Ti_peak': Ti_values, 'ni_peak': ni_values}
plasmaPropertiesResults = run_sweep(name, variables=variables, reset=False)

fig, ax = plt.subplots()
for ni_peak, grp in plasmaPropertiesResults.groupby('ni_peak'):
    # Convert density to m^-3
    grp.plot(ax=ax, x='Ti_peak', y='mean', label=f'$n_i$={ni_peak*1e6:.1e} (m$^{{-3}}$)')
    ax.fill_between(grp['Ti_peak'], grp['mean'] - grp['std. dev.'], grp['mean'] + grp['std. dev.'], alpha=0.2)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('$T_{i, \mathrm{peak}}$ (keV)')
ax.set_ylabel('Counts/s')
ax.legend()
plt.savefig(f'{figures_folder}/neutronics_plasma_sweep.png', dpi=600)
plt.show()