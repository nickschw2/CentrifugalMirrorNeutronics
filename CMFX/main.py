from CMFX_source import *
from constants import *
from param_sweep import *

import sys
sys.path.append('../')
from plot_style import *

# Options for temperature and density profiles are either 'parabolic' or 'Huang2004'
profile = 'parabolic'

### STANDARD SIMULATION ###
# path = f'{root}/{examples_folder}/standard'
# reset = True
# if reset:
#     if os.path.isdir(path):
#         shutil.rmtree(path)
#     source = CMFX_Source(profile=profile, create_mesh_tally=True)
#     source.run(path)
#     results = source.read_results()
#     results.to_csv(f'{path}/results_{profile}.csv')
# else:
#     try:
#         result = pd.read_csv(f'{path}/results_{profile}.csv')
#         with open(f'{path}/{source_pkl}', 'rb') as file:
#             source = pickle.load(file)
#     except FileNotFoundError as error:
#         print(error)
#         print('Results file does not yet exist. Please set "reset" to True.')

# source.plot_flux()
# breakpoint()

### CONVERGENCE SWEEP ###
N_particles = np.logspace(3, 6, 16).astype('int')
name = f'convergence_{profile}'
variables = {'particles': N_particles}
convergenceResults = run_sweep(name, variables=variables, reset=False)
convergenceResults['MCSE'] = convergenceResults['std. dev.'] / convergenceResults['mean']

convergenceResults.plot(x='particles', y='MCSE', logx=True, logy=False)
plt.xlabel('# Particles')
plt.ylabel(r'Coeff. of Variance ($\mu$ / $\sigma$)')
plt.savefig(f'{figures_folder}/neutronics_convergence.png', dpi=600)
plt.close()
# plt.show()

### DETECTOR LOCATION SWEEP ###
radialDistances = np.linspace(50, 90, 17)
axialDistances = np.linspace(0, 30, 5)
name = f'detector_location_{profile}'
variables = {'radialDistance': radialDistances, 'axialDistance': axialDistances}
detectorLocationResults = run_sweep(name, profile=profile, variables=variables, reset=False)

fig, ax = plt.subplots()
for axialDistance, grp in detectorLocationResults.groupby('axialDistance'):
    grp.plot(ax=ax, x='radialDistance', y='mean', label=f'z={int(axialDistance)} (cm)')
    ax.fill_between(grp['radialDistance'], grp['mean'] - grp['std. dev.'], grp['mean'] + grp['std. dev.'], alpha=0.2)

ax.set_xlabel('Radial Distance (cm)')
ax.set_ylabel('Counts/s')
ax.legend()
ax.set_ylim(bottom=0)
plt.savefig(f'{figures_folder}/neutronics_location_sweep_{profile}.png', dpi=600)
plt.close()
# plt.show()

### TEMPERATURE AND DENSITY SWEEP ###
Ti_values = np.logspace(np.log10(0.2), 1, 15) # keV
ni_values = np.logspace(12, 14, 5) # cm^-3
name = f'plasma_properties_{profile}'
variables = {'Ti_avg': Ti_values, 'ni_avg': ni_values}
plasmaPropertiesResults = run_sweep(name, profile=profile, variables=variables, reset=False)

fig, ax = plt.subplots()
for ni_avg, grp in plasmaPropertiesResults.groupby('ni_avg'):
    # Convert density to m^-3
    grp.plot(ax=ax, x='Ti_avg', y='mean', label=f'$n_i$={ni_avg*1e6:.1e} (m$^{{-3}}$)')
    ax.fill_between(grp['Ti_avg'], grp['mean'] - grp['std. dev.'], grp['mean'] + grp['std. dev.'], alpha=0.2)

ax.set_xscale('log')
ax.set_xlim(left=1e-1)
ax.set_yscale('log')
ax.set_xlabel(r'$T_{i, \mathrm{avg}}$ (keV)')
ax.set_ylabel('Counts/s')
ax.legend()
plt.savefig(f'{figures_folder}/neutronics_plasma_sweep_{profile}.png', dpi=600)
plt.close()
# plt.show()

### Interpolate plasma properties for neutron count threshold ###
threshold = 100 #n/s
densities = [1e12, 1e13] # cm^-3
for density in densities:
    df = plasmaPropertiesResults[plasmaPropertiesResults['ni_avg'] == density]
    Ti_threshold = np.interp(threshold, df['mean'], df['Ti_avg'])
    print(Ti_threshold)
