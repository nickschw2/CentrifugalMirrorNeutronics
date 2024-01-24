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
    distance = calibration_results['Distance']
    mean = calibration_results['Mean'] / duration
    std = calibration_results['Std'] / duration
    added_distance = calibration_results['Added Distance (cm)'][0]
    activity = calibration_results['Activity (n/s)'][0]
    shielding = True if calibration_results['Shielding'][0] == 1 else False
    distance = distance + added_distance

    return distance, mean, std, activity, shielding

# The run conditions for HE3DET01 and HE3DET02 are different
# HE3DET02 was with Pb shielding and aluminum enclosure
# Need to convert distance to actual distance to center of tube
distance, mean, std, activity, shielding = get_calibration_data(DET_NUM)

### MODEL VARIABLE ###
# distances = np.linspace(5, 40, 36)
distances = np.linspace(min(distance), max(distance), 20)
name = f'HE3DET0{DET_NUM}'
variables = {'distance': distances, 'activity': [activity], 'shielding': [shielding]}
results = run_sweep(name, variables=variables, reset=reset)

fig, ax = plt.subplots()
ax.plot(distance, mean, label=f'Exp. Det. {DET_NUM}')
ax.fill_between(distance, mean - std, mean + std, alpha=0.2)
ax.plot(results['distance'], results['mean'], label=f'OpenMC Det. {DET_NUM}')
ax.fill_between(results['distance'], results['mean'] - results['std. dev.'], results['mean'] + results['std. dev.'], alpha=0.2)

ax.set_xlabel('Distance (cm)')
ax.set_ylabel('Counts/sec')
ax.legend()
plt.show()

### PLOT BOTH CALIBRATIONS ###
if not reset:
    distance_01, mean_01, std_01, activity_01, shielding_01 = get_calibration_data(1)
    distance_02, mean_02, std_02, activity_02, shielding_02 = get_calibration_data(2)
    results_01 = run_sweep('HE3DET01')
    results_02 = run_sweep('HE3DET02')

    fig, ax = plt.subplots()
    ax.plot(distance_01, mean_01, label='Exp. Det. 1')
    ax.fill_between(distance_01, mean_01 - std_01, mean_01 + std_01, alpha=0.2)
    ax.plot(results_01['distance'], results_01['mean'], label='OpenMC Det. 1')
    ax.fill_between(results_01['distance'], results_01['mean'] - results_01['std. dev.'], results_01['mean'] + results_01['std. dev.'], alpha=0.2)

    ax.plot(distance_02, mean_02, label='Exp. Det. 2')
    ax.fill_between(distance_02, mean_02 - std_02, mean_02 + std_02, alpha=0.2)
    ax.plot(results_02['distance'], results_02['mean'], label='OpenMC Det. 2')
    ax.fill_between(results_02['distance'], results_02['mean'] - results_02['std. dev.'], results_02['mean'] + results_02['std. dev.'], alpha=0.2)

    ax.set_xlabel('Distance (cm)')
    ax.set_ylabel('Counts/sec')
    ax.legend()
    plt.show()




