from point_source import *
from constants import *
from param_sweep import *
import sys
sys.path.append('../')
from plot_style import *

reset = False
### COMPARISON TO ACTUAL CALIBRATION DATA ###
def get_calibration_data(det_num):
    CALIBRATION_FILE = f'NEUTRON_CALIBRATION_DATA_CYLINDRICAL_HE3DET0{det_num}.XLSX'
    calibration_results = pd.read_excel(CALIBRATION_FILE)
    distance = calibration_results['Distance'].dropna()
    mean = calibration_results['Mean'].dropna() / duration
    std = calibration_results['Std'].dropna() / duration
    background_mean = mean[distance == 'BACKGROUND'].iloc[0]
    background_std = std[distance == 'BACKGROUND'].iloc[0]
    added_distance = calibration_results['Added Distance (cm)'][0]
    activity = calibration_results['Activity (n/s)'][0]
    shielding = True if calibration_results['Shielding'][0] == 1 else False
    distance = distance.iloc[:-1].astype(np.float64) + added_distance

    # Subtract background
    mean = mean.iloc[:-1] - background_mean
    std = std.iloc[:-1] + background_std # Add uncertainty propogation

    return distance, mean, std, activity, shielding

def get_efficiency(det_num):
    distance, mean, std, activity, shielding = get_calibration_data(det_num)
    # Find the solid angle subtended by rectangle at a given normal distance from center of sphere (https://www.researchpublish.com/upload/book/HCRS%20THEORY%20OF%20POLYGON-663.pdf)
    Omega = 4 * np.arcsin(He3_diameter * He3_length / np.sqrt((He3_length**2 + 4 * distance**2) * (He3_diameter**2 + 4 * distance**2)))
    # Omega = 4 * np.arcsin(HDPE_diameter * HDPE_height / np.sqrt((HDPE_height**2 + 4 * distance**2) * (HDPE_diameter**2 + 4 * distance**2)))
    # Omega = He3_diameter * He3_length / distance**2
    flux = activity * Omega / (4 * np.pi)
    efficiency = mean / flux
    efficiency_std = std / flux

    return efficiency, efficiency_std

def get_model_efficiency(det_num):
    name = f'HE3DET0{det_num}'
    results = pd.read_csv(f'{root}/{sweep_folder}/{name}/results.csv')
    absorption_results = results[results['score'] == 'absorption']
    flux_results = results[results['score'] == 'flux']
    distance = absorption_results['distance']

    # Normalize the flux
    # We've already normalized the strength of the source to the activity of the source
    # Originally in units of particle-cm/sec
    # Divide by volume and multiply by area to get particle/sec:
    flux_results.loc[flux_results['score'] == 'flux', 'mean'] *= 4 / (np.pi * He3_diameter)
    flux_results.loc[flux_results['score'] == 'flux', 'std. dev.'] *= 4 / (np.pi * He3_diameter)

    # Model efficiency
    efficiency = absorption_results['mean'].to_numpy() / flux_results['mean'].to_numpy()
    # Use formula for error propogation for division
    efficiency_std = efficiency * np.sqrt((flux_results['std. dev.'].to_numpy() / flux_results['mean'].to_numpy())**2 + (absorption_results['std. dev.'].to_numpy() / absorption_results['mean'].to_numpy())**2)

    return distance, efficiency, efficiency_std


# The run conditions for HE3DET01 and HE3DET02 are different
# HE3DET02 was with Pb shielding and aluminum enclosure
# Need to convert distance to actual distance to center of tube
distance, mean, std, activity, shielding = get_calibration_data(DET_NUM)

### MODEL VARIABLE ###
distances = np.linspace(min(distance), max(distance), 20)
name = f'HE3DET0{DET_NUM}'
variables = {'distance': distances, 'activity': [activity], 'shielding': [shielding]}
results = run_sweep(name, variables=variables, reset=reset)
absorption_results = results[results['score'] == 'absorption']
print(f'Model Det. {DET_NUM} Max Error: {max(absorption_results["std. dev."] / absorption_results["mean"])}')


# Find relative error between the two
error = np.abs(mean - np.interp(distance, absorption_results['distance'], absorption_results['mean'])) / mean * 100
max_error = max(error)
avg_error = np.mean(error)
print(f'Average Error (%): {avg_error}')

fig, ax = plt.subplots()
ax.plot(distance, mean, label=f'Exp. Det. {DET_NUM}')
ax.fill_between(distance, mean - std, mean + std, alpha=0.2)
ax.plot(absorption_results['distance'], absorption_results['mean'], label=f'OpenMC Det. {DET_NUM}')
ax.fill_between(absorption_results['distance'], absorption_results['mean'] - absorption_results['std. dev.'], absorption_results['mean'] + absorption_results['std. dev.'], alpha=0.2)

ax.set_xlabel('Distance (cm)')
ax.set_ylabel('Counts/sec')
ax.legend()
fig.savefig(f'HE3DET0{DET_NUM}_comparison.png', dpi=300)
plt.show()

### PLOT BOTH CALIBRATIONS ###
distance_01, mean_01, std_01, activity_01, shielding_01 = get_calibration_data(1)
distance_02, mean_02, std_02, activity_02, shielding_02 = get_calibration_data(2)
print(f'Det 1 Exp. Max Error: {max(std_01 / mean_01)}')
print(f'Det 2 Exp. Max Error: {max(std_02 / mean_02)}')

fig, ax = plt.subplots()
ax.plot(distance_01, mean_01 / activity_01, label='Exp. Det. 1')
ax.fill_between(distance_01, (mean_01 - std_01) / activity_01, (mean_01 + std_01) / activity_01, alpha=0.2)

ax.plot(distance_02, mean_02 / activity_02, label='Exp. Det. 2')
ax.fill_between(distance_02, (mean_02 - std_02) / activity_02, (mean_02 + std_02) / activity_02, alpha=0.2)

ax.set_xlabel('Distance (cm)')
ax.set_ylabel('Counts/sec/activity (-)')
ax.legend()
fig.savefig(f'calibration_comparison.png', dpi=300)
plt.show()

### PLOT FLUX ###
fig, ax = plt.subplots()
# Actual efficiency
efficiency_01, efficiency_std_01 = get_efficiency(1)
efficiency_02, efficiency_std_02 = get_efficiency(2)

distance_01, efficiency_01, efficiency_std_01 = get_model_efficiency(1)
distance_02, efficiency_02, efficiency_std_02 = get_model_efficiency(2)

# ax.plot(distance_01, efficiency_01, label=f'Exp. Det. 1')
# ax.fill_between(distance_01, efficiency_01 - efficiency_std_01, efficiency_01 + efficiency_std_01, alpha=0.2)
# ax.plot(distance_02, efficiency_02, label=f'Exp. Det. 2')
# ax.fill_between(distance_02, efficiency_02 - efficiency_std_02, efficiency_02 + efficiency_std_02, alpha=0.2)
ax.plot(distance_01, efficiency_01, label=f'OpenMC Det. 1')
ax.fill_between(distance_01, efficiency_01 - efficiency_std_01, efficiency_01 + efficiency_std_01, alpha=0.2)
ax.plot(distance_02, efficiency_02, label=f'OpenMC Det. 2')
ax.fill_between(distance_02, efficiency_02 - efficiency_std_02, efficiency_02 + efficiency_std_02, alpha=0.2)

ax.set_xlabel('Distance (cm)')
ax.set_ylabel('Detector Efficiency')
ax.legend()
ax.set_ylim(bottom=0, top=0.25)
fig.savefig(f'HE3DET_efficiency.png', dpi=300)
plt.show()

### EXAMPLE FLUX PLOTS ###
reset = False
def get_flux_results(folder, distance=8, moderator=True, reset=False):
    path = f'{root}/{examples_folder}/{folder}'
    if reset:
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.mkdir(path)
        source = Point_Source(activity=activity_02, shielding=shielding_02, distance=distance, moderator=moderator, create_mesh_tally=True)
        source.run(path)
        results = source.read_results()
        results.to_csv(f'{path}/results.csv')
    else:
        try:
            results = pd.read_csv(f'{path}/results.csv')
            with open(f'{path}/{source_pkl}', 'rb') as file:
                source = pickle.load(file)
        except FileNotFoundError as error:
            print(error)
            print('Results file does not yet exist. Please set "reset" to True.')

    return source

scenarios = {'close_noModerator': {'distance': 8,
                                   'moderator': False,
                                   'reset': reset},
             'close_moderator': {'distance': 8,
                                 'moderator': True,
                                 'reset': reset},
             'far_noModerator': {'distance': 20,
                                 'moderator': False,
                                 'reset': reset},
             'far_Moderator': {'distance': 20,
                               'moderator': True,
                               'reset': reset},}

# 2 x 2 plots
# fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(5, 8))
# for i, (scenario, kwargs) in enumerate(scenarios.items()):
#     source = get_flux_results(scenario, **kwargs)
#     Z, Y, flux = source.get_flux_map()
#     im = axes[i % 2, i // 2].contourf(Z, Y, flux)
#     axes[i % 2, i // 2].set_title(f'{kwargs["distance"]} cm, {"w/" if kwargs["moderator"] else "no"} mod.')
#     # axes[i % 2, i // 2].set_aspect('equal', adjustable='box')
#     fig.colorbar(im, ax=axes[i % 2, i // 2], label=r'Flux $\left( \frac{\#}{ \mathrm{cm}^2 \mathrm{s} } \right)$')

# fig.text(0.5, -0.04, 'Radius (cm)', ha='center')
# fig.text(-0.04, 0.5, 'Height (cm)', va='center', rotation='vertical')
# plt.savefig(f'{root}/{examples_folder}/flux_comparison.png', dpi=300, bbox_inches='tight')
# plt.show()

# 4 x 1 plots
fig, axes = plt.subplots(1, 4, sharex=True, sharey=True, figsize=(8, 4))
for i, (scenario, kwargs) in enumerate(scenarios.items()):
    source = get_flux_results(scenario, **kwargs)
    Z, Y, flux = source.get_flux_map()
    im = axes[i].contourf(Z, Y, flux)
    axes[i].set_title(f'{kwargs["distance"]} cm, \n {"w/" if kwargs["moderator"] else "no"} mod.')
    # axes[i % 2, i // 2].set_aspect('equal', adjustable='box')
    fig.colorbar(im, ax=axes[i], label=r'Flux $\left( \frac{\#}{ \mathrm{cm}^2 \mathrm{s} } \right)$')

fig.text(0.5, -0.04, 'Radius (cm)', ha='center')
fig.text(-0.04, 0.5, 'Height (cm)', va='center', rotation='vertical')
plt.savefig(f'{root}/{examples_folder}/calibration_flux_comparison.png', dpi=300, bbox_inches='tight')
plt.show()