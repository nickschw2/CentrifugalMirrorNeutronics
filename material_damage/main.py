# from constants import *
from param_sweep import *
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import sys
sys.path.append('../')
from plot_style import *

### CONVERGENCE SWEEP ###
# N_particles = np.logspace(3, 6, 10).astype('int')
# name = 'convergence'
# variables = {'particles': N_particles}
# convergenceResults = run_sweep(name, variables=variables, reset=True)
# convergenceResults['MCSE'] = convergenceResults['std. dev.'] / convergenceResults['mean']

# convergenceResults.plot(x='particles', y='MCSE', logx=True, logy=False)
# plt.xlabel('# Particles')
# plt.ylabel(r'Coeff. of Variance ($\mu$ / $\sigma$)')
# plt.savefig(f'{figures_folder}/neutronics_convergence.png', dpi=600)
# plt.show()

### STANDARD EXAMPLE ###
directory = f'{root}/standard'
source = MaterialDamageSource(material_name='SiC')
# source.run(directory)
# source.plot_flux_map()

### PARAMETER SWEEP ###
name = 'sweep'
materials = ['BN', 'SiC', 'W']
fuels = ['D-D', 'D-T']
variables = {'fuel': fuels, 'material_name': materials}
sweepResults = run_sweep(name, variables=variables, reset=False)
z = np.linspace(0, target_thickness, N_mesh_points)

# Normalize all results to 1
dpa_max = np.max(np.vstack(sweepResults['dpa'].to_numpy()))
abs_max = np.max(np.vstack(sweepResults['abs'].to_numpy()))
flux_max = np.max(np.vstack(sweepResults['flux'].to_numpy()))

# Make stacked plots
fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
for (material, fuel), grp in sweepResults.groupby(['material_name', 'fuel']):
    color = colors['qualitative'][materials.index(material)]
    linestyle = line_styles[fuels.index(fuel)]

    # dpa plot
    axs[0].plot(z, grp['dpa'].to_numpy()[0] / dpa_max, label=f'{material}, {fuel}', color=color, linestyle=linestyle)
    axs[1].plot(z, grp['abs'].to_numpy()[0] / abs_max, label=f'{material}, {fuel}', color=color, linestyle=linestyle)
    axs[2].plot(z, grp['flux'].to_numpy()[0] / flux_max, label=f'{material}, {fuel}', color=color, linestyle=linestyle)

    axs[0].set_ylabel('dpa (Norm.)')
    axs[1].set_ylabel('Absorption (Norm.)')
    axs[2].set_ylabel('Flux (Norm.)')

axs[-1].set_xlabel('Depth (cm)')
plt.xlim(right=80)
plt.legend()
plt.savefig('neutron_materials_damage.png', dpi=300)
plt.show()

# Make bar plot showing the comprehesive results
metrics = ['depth', 'dpa_tot', 'abs_tot']
# Set width of bars
bar_width = 0.25

# Set positions of the bars on X axis
r = np.arange(len(metrics))
# Combine all data
df = sweepResults[['material_name', 'fuel', 'depth', 'dpa_tot', 'abs_tot']]
# Normalize each metric to the max sum of the fuels
for metric in metrics:
    max_sum = df.groupby('material_name')[metric].sum().max()
    df[metric] = df[metric].div(max_sum)

# Create bars
fig, ax = plt.subplots(figsize=(8, 4))
# Create bars
for i, material in enumerate(materials):
    for j, fuel in enumerate(fuels):
        color = colors['qualitative'][i]
        values = df[(df['material_name'] == material) & (df['fuel'] == fuel)][metrics].values[0]
        x = r + i * bar_width
        if fuel == 'D-D':
            bars = ax.bar(x, values, width=bar_width, color=color, edgecolor='white', linewidth=1)
            bottom = values
        else:  # D-T
            bars = ax.bar(x, values, width=bar_width, bottom=bottom, color=color, edgecolor='white', linewidth=1)
            for bar in bars:
                bar.set_hatch('////')
                bar.set_edgecolor('white')
    
# Add xticks on the middle of the group bars
ax.set_xticks(r + bar_width)
labels = ['Depth', 'Dpa tot.', 'Absorption tot.']
ax.set_xticklabels(labels, fontweight='bold')

# Create legend for metrics
materials_handles = [Patch(facecolor=colors['qualitative'][i], edgecolor='white', label=label) 
                 for i, label in enumerate(materials)]
materials_legend = ax.legend(handles=materials_handles, loc='upper left')

# Create legend for fuels
fuel_handles = [Patch(facecolor='black', edgecolor='white', label='D-D'),
               Patch(facecolor='black', hatch='////', edgecolor='white', label='D-T')]
ax.add_artist(ax.legend(handles=fuel_handles, loc='upper right', bbox_to_anchor=(0.42, 1.0)))
ax.add_artist(materials_legend)

# Move grid to background
ax.set_axisbelow(True)

ax.set_ylabel('Normalized Performance')

plt.savefig('neutron_materials_damage_summary.png', dpi=300)
plt.show()