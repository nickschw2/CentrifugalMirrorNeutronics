import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
import numpy as np
import os
from constants import *

import sys
sys.path.append('../')
from plot_style import *

# Set the peak values of the temperature and density at the midplane
Ti_avg = 1 # [keV]
ni_avg = 1e13 # [cm^-3]

# The purpose of this file is to produce a map of temperature and density with
# respect to r and z
if 'MagB_v1.csv' not in os.listdir('.'):
    data = pd.read_csv('MagB_v1.txt', header=1, usecols=[0, 2, 3])
    # Convert from m to cm
    data[['X', 'Z']] = data[['X', 'Z']] * 100
    # Set the spatial limit of interest
    data = data[(data['X'] <= plasma_outerRadius) & (data['Z'] >= -plasma_length / 2)]
    r = data['X']
    z = data['Z']
    # Get unique values of r and z
    rvals = data['X'].unique()
    zvals = data['Z'].unique()
    mag = data['Magnitude']
    data['Flux'] = mag * np.pi * r**2

    data.to_csv('MagB_v1.csv', index=False)
else:
    data = pd.read_csv('MagB_v1.csv')
    
    # Get unique values of r and z
    rvals = data['X'].unique()
    zvals = data['Z'].unique()


# Define the flux lines to follow based on the midplane
midplane = data[data['Z'] == 0]
midplane_radius = np.linspace(plasma_innerRadius, plasma_outerRadius, N_points)
flux_midplane_interp = np.interp(midplane_radius, midplane['X'], midplane['Flux'])

# Reshape the flux values to a 2D array that corresponds to r and z
fluxvals = data['Flux'].values.reshape(len(rvals), len(zvals))
# The contour plot returns values of the contours that can be used to track the inner and outer radii
cs = plt.contour(zvals, rvals, fluxvals, flux_midplane_interp)
plt.xlabel('z (cm)')
plt.ylabel('r (cm)')
cb = plt.colorbar()
cb.ax.set_title('$B_{mag}$ Flux\n(T m$^{2}$)')
plt.savefig('figures/Bmag_flux.png', dpi=300)
# plt.show()
plt.close()

# Find the inner, middle, and outer radius of the plasma along z
# Note that the average value of the inner and outer is *almost* in agreement with the middle radius
inside = cs.collections[0].get_paths()[0].vertices
z_inner, r_inner = (inside[:, 0], inside[:, 1])
middle = cs.collections[int((N_points - 1) / 2)].get_paths()[0].vertices
z_mid, r_mid = (middle[:, 0], middle[:, 1])
outside = cs.collections[-1].get_paths()[0].vertices
z_outer, r_outer = (outside[:, 0], outside[:, 1])
# Need to sort z_mid
z_mid.sort()
z_mid = np.unique(z_mid)

# The length of the arrays for the contour lines are different, so we interpolate on the middle one
# to make it the same length
innerRadius = np.interp(z_mid, z_inner, r_inner)
outerRadius = np.interp(z_mid, z_outer, r_outer)

# Make a 2D array over the entire inner flux surface to outer flux surface
radius = np.linspace(min(innerRadius), max(outerRadius), N_points)
R, Z = np.meshgrid(radius, z_mid)

# Need to use results from MCTrans++ to determine what the appropriate Mach number is for a given Ti and ni
file = 'MCTrans_results.csv'
results = pd.read_csv(file)
electronDensity = np.unique(results['electronDensity'].to_numpy()) / 1e6
electronTemperature = np.unique(results['electronTemperature'].to_numpy())
Mach = results['Mach'].values.reshape(len(electronDensity), len(electronTemperature))
f_Mach = interpolate.RectBivariateSpline(electronDensity, electronTemperature, Mach)

# Load results from Huang2004 to create function for profile
density_file = 'density_Huang2004.csv'
temperature_file = 'temperature_Huang2004.csv'
density_Huang2004 = pd.read_csv(density_file, names=['x', 'y'])
temperature_Huang2004 = pd.read_csv(temperature_file, names=['x', 'y'])
density_Huang2004 = density_Huang2004.sort_values('x')
temperature_Huang2004 = temperature_Huang2004.sort_values('x')
# Normalize profiles to inner and outer radius at midplane and to max value of 1
density_Huang2004['y'] = density_Huang2004['y'] / max(density_Huang2004['y'])
temperature_Huang2004['y'] = temperature_Huang2004['y'] / max(temperature_Huang2004['y'])
density_Huang2004['x'] = np.interp(density_Huang2004['x'], [density_Huang2004['x'].min(), density_Huang2004['x'].max()], [plasma_innerRadius, plasma_outerRadius])
temperature_Huang2004['x'] = np.interp(temperature_Huang2004['x'], [temperature_Huang2004['x'].min(), temperature_Huang2004['x'].max()], [plasma_innerRadius, plasma_outerRadius])


def get_profiles(Ti_avg, ni_avg, profile='parabolic', plot=False):
    Ti = np.zeros((len(z_mid), len(radius)))
    ni = np.zeros((len(z_mid), len(radius)))

    M = f_Mach(ni_avg, Ti_avg) # Assume some Mach number based on results from MCTrans++
    # Calculate the temperature and density profile for a given value of z
    for i, z in enumerate(z_mid):
        # Assume that the temperature and density both have parabolic profiles at a given value of z
        mask = (radius >= innerRadius[i]) & (radius <= outerRadius[i])
        T_r_profile = np.zeros(len(radius))
        n_r_profile = np.zeros(len(radius))
        if profile == 'parabolic':
            T_r_profile[mask] = 1 - (((outerRadius[i] + innerRadius[i])/2 - radius[mask]) / ((outerRadius[i] - innerRadius[i])/2))**2
            n_r_profile[mask] = T_r_profile[mask]
        elif profile == 'Huang2004':
            T_indices = np.linspace(temperature_Huang2004['x'].min(), temperature_Huang2004['x'].max(), len(radius[mask]))
            n_indices = np.linspace(density_Huang2004['x'].min(), density_Huang2004['x'].max(), len(radius[mask]))
            T_r_profile[mask] = np.interp(T_indices, temperature_Huang2004['x'], temperature_Huang2004['y'])
            n_r_profile[mask] = np.interp(n_indices, density_Huang2004['x'], density_Huang2004['y'])

        # All values less than zero set to zero
        n_r_profile[n_r_profile < 0] = 0
        T_r_profile[T_r_profile < 0] = 0
        # To set the average value, need to divide by the radially weighted average
        T_avg_factor = np.average(T_r_profile[mask], weights=radius[mask])
        n_avg_factor = np.average(n_r_profile[mask], weights=radius[mask])
        Ti[i, :] = (Ti_avg / T_avg_factor) * T_r_profile

        # Calculate the density based on equation 2.9 and 2.20 in Schwartz et. al. 2024 (MCTrans++ Paper)
        tau = 1 # Assume Te=Ti
        Z_i = 1 # Assume deuterium plasma
        Chi_i = tau / (Z_i + tau) * M**2 / 2 * (1 - (r_mid[i] / r_mid[-1])**2) # Value of Chi is based of the flux line in the center of the plasma
        ni_max = ni_avg * np.exp(-Chi_i) # Equation 2.9
        ni[i, :] = (ni_max / n_avg_factor) * n_r_profile

    if plot:
        # Contour plot of temperature
        plt.contourf(Z, R, Ti, np.linspace(1e-6, np.max(Ti), 100))
        plt.xlabel('z (cm)')
        plt.ylabel('r (cm)')
        cb = plt.colorbar(ticks=np.linspace(0, Ti_avg, 11))
        cb.ax.set_title('$T_i$ (keV)')

        plt.savefig(f'figures/Ti_contour_{profile}.png', dpi=300)
        plt.show()

        # Contour plot of density
        plt.contourf(Z, R, ni, np.linspace(1e-6, np.max(ni), 100))
        plt.xlabel('z (cm)')
        plt.ylabel('r (cm)')
        cb = plt.colorbar(ticks=np.linspace(0, ni_avg, 11))
        cb.ax.set_title('$n_i$ (cm$^{-3}$)')
        plt.savefig(f'figures/ni_contour_{profile}.png', dpi=300)
        plt.show()

    # Need to return z both positive and negative
    # Basically reflect values around z=0
    z = np.concatenate((z_mid, z_mid[::-1][1:] * -1))
    Ti = np.concatenate((Ti, Ti[::-1][1:]))
    ni = np.concatenate((ni, ni[::-1][1:]))
    return radius, z, Ti, ni

get_profiles(Ti_avg, ni_avg, profile='Huang2004', plot=False)