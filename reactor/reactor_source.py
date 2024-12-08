from constants import *
import openmc
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import neutronics_material_maker as nmm
import pickle
from scipy import integrate, interpolate

class CMFX_Source():
    def __init__(self, Ti_peak=10, ni_peak=1e14, particles=250000, radialDistance=50, axialDistance=0,
                 fuel='DT', create_mesh_tally=False):
        self.Ti_peak = Ti_peak
        self.ni_peak = ni_peak
        self.particles = particles
        self.radialDistance = radialDistance
        self.axialDistance = axialDistance
        self.fuel = fuel
        self.create_mesh_tally = create_mesh_tally

        # Determine whether the plasma is too cold to generate any neutrons, which would lead to runtime error in simulation
        self.neutrons_generated = self.Ti_peak >= Ti_min

        self.set_materials()
        self.set_geometry()
        self.set_source()
        self.run_settings()
        self.set_tallies()

    def set_materials(self):
        self.Al_material = nmm.Material.from_library(name='Aluminum, alloy 6061-O').openmc_material
        self.Steel_material = nmm.Material.from_library(name='Steel, Stainless 316L').openmc_material

        if self.fuel == 'DD':
            self.plasma_material = nmm.Material.from_library(name='DD_plasma').openmc_material
            self.plasma_material.set_density("g/cm3", self.ni_peak * 2 * ProtonMass)
        elif self.fuel == 'DT':
            self.plasma_material = nmm.Material.from_library(name='DT_plasma').openmc_material
            self.plasma_material.set_density("g/cm3", self.ni_peak * 2.5 * ProtonMass)
        
        self.BN_material = nmm.Material(name='Boron Nitride (BN)', **materials['Boron Nitride (BN)']).openmc_material

        self.materials = openmc.Materials([self.Al_material, self.Steel_material, self.plasma_material,
                                           self.BN_material])

    def set_geometry(self):
        ### SURFACES ###
        # Void area
        void_surface = openmc.model.RightCircularCylinder((0, 0, -void_length / 2), void_length, void_radius,
                                                          axis='z', boundary_type='vacuum')
        
        # Plasma Surfaces
        plasma_outerSurface = openmc.model.RightCircularCylinder((0, 0, -plasma_length / 2), plasma_length,
                                                                 plasma_outerRadius, axis='z')
        plasma_innerSurface = openmc.model.RightCircularCylinder((0, 0, -plasma_length / 2), plasma_length,
                                                                 plasma_innerRadius, axis='z')
        
        # Center conductor surfaces
        centerConductor_outerSurface = openmc.model.RightCircularCylinder((0, 0, -centerCondutor_length / 2),
                                                                          centerCondutor_length, centerCondutor_radius, axis='z')
        centerConductor_innerSurface = openmc.model.RightCircularCylinder((0, 0, -centerCondutor_length / 2),
                                                                          centerCondutor_length, centerCondutor_radius - centerCondutor_thickness, axis='z')
        
        # Chamber surfaces
        chamber_outerSurface = openmc.model.RightCircularCylinder((0, 0, -chamber_length / 2),
                                                                  chamber_length, chamber_radius, axis='z')
        chamber_innerSurface = openmc.model.RightCircularCylinder((0, 0, -chamber_length / 2),
                                                                  chamber_length, chamber_radius - chamber_thickness, axis='z')
        
        # Insulator surfaces
        insulator_outerSurface = openmc.model.RightCircularCylinder((0, 0, -insulator_length - insulator_axialStart),
                                                                  insulator_length, insulator_outerRadius, axis='z')
        insulator_innerSurface = openmc.model.RightCircularCylinder((0, 0, -insulator_length - insulator_axialStart),
                                                                  insulator_length, insulator_outerRadius - insulator_thickness, axis='z')
        insulator_outerPlateSurface = openmc.model.RightCircularCylinder((0, 0, -insulator_length - insulator_axialStart - insulator_thickness),
                                                                  insulator_thickness, insulator_outerRadius, axis='z')
        insulator_innerPlateSurface = openmc.model.RightCircularCylinder((0, 0, -insulator_length - insulator_axialStart - insulator_thickness),
                                                                  insulator_thickness, insulator_innerRadius, axis='z')

        # CELLS
        plasma_region = (-plasma_outerSurface & +plasma_innerSurface)
        centerConductor_region = (-centerConductor_outerSurface & +centerConductor_innerSurface)
        chamber_region = (-chamber_outerSurface & +chamber_innerSurface)
        insulator_region = (-insulator_outerSurface & +insulator_innerSurface) & (-insulator_outerPlateSurface & +insulator_innerPlateSurface)
        void_region = -void_surface & ~plasma_region & ~centerConductor_region & ~chamber_region & ~insulator_region

        self.void_cell = openmc.Cell(region=void_region)
        self.plasma_cell = openmc.Cell(region=plasma_region)
        self.centerConductor_cell = openmc.Cell(region=centerConductor_region)
        self.chamber_cell = openmc.Cell(region=chamber_region)
        self.insulator_cell = openmc.Cell(region=chamber_region)

        self.plasma_cell.fill = self.plasma_material
        self.centerConductor_cell.fill = self.Steel_material
        self.chamber_cell.fill = self.Al_material
        self.insulator_cell.fill = self.BN_material

        self.universe = openmc.Universe(cells=[self.void_cell, self.centerConductor_cell, self.chamber_cell,
                                               self.plasma_cell, self.insulator_cell])
        self.geometry = openmc.Geometry(root=self.universe)

    def get_neutron_profile(self):
        # Arbitrarily assume a parabolic plasma temperature and density
        # Create T(r, z) and n(r, z)
        N_points = 1000
        r = np.linspace(plasma_innerRadius, plasma_outerRadius, N_points)
        z = np.linspace(-plasma_length / 2, plasma_length / 2, N_points)
        R, Z = np.meshgrid(r, z)
        r_profile = 1 - (((plasma_outerRadius + plasma_innerRadius)/2 - R) / ((plasma_outerRadius - plasma_innerRadius)/2))**2
        z_profile = 1 - (Z / (plasma_length / 2))**2
        Ti_profile = self.Ti_peak * r_profile * z_profile
        ni_profile = self.ni_peak * r_profile * z_profile

        # Create interpolation functions so we can pass them to the integral later
        f_Ti = interpolate.RectBivariateSpline(r, z, Ti_profile)
        f_ni = interpolate.RectBivariateSpline(r, z, ni_profile)
        
        if self.fuel == 'DD':
            # Parameters are taken as the D(d, n)3He reaction fitting parameters from table VII of: https://iopscience.iop.org/article/10.1088/0029-5515/32/4/I07/pdf
            B_G = 31.3970 # keV^(1/2)
            mrcc = 937814 # Actually m_r*c^2, keV
            C1 = 5.43360E-12 # 1/keV
            C2 = 5.85778E-3 # 1/keV
            C3 = 7.68222E-3 # 1/keV
            C4 = 0 # 1/keV
            C5 = -2.96400E-6 # 1/keV
            C6 = 0 # 1/keV
            C7 = 0 # 1/keV
        elif self.fuel == 'DT':
            # Parameters are taken as the T(d, n)4He reaction fitting parameters from table VII of: https://iopscience.iop.org/article/10.1088/0029-5515/32/4/I07/pdf
            B_G = 34.3827 # keV^(1/2)
            mrcc = 1124656 # Actually m_r*c^2, keV
            C1 = 1.17302e-9 # 1/keV
            C2 = 1.51361e-2 # 1/keV
            C3 = 7.51886e-2 # 1/keV
            C4 = 4.60643e-3 # 1/keV
            C5 = 1.35000e-2 # 1/keV
            C6 = -1.06750e-4 # 1/keV
            C7 = 1.36600e-5 # 1/keV

        # Eqns 12-14 of above paper
        # Returns <sigma * v> in cm^3 / s
        # We also multiply by density^2 to get units of cm^-3 s^-1
        # Need this to be a function of r and z so that we can integrate over it and get the total neutrons
        def get_reaction_rate(r, z):
            Ti_values = f_Ti(r, z)
            ni_values = f_ni(r, z)
            sigma_v_values = np.zeros(Ti_values.shape)

            for i, j in np.ndindex(Ti_values.shape):
                Ti = Ti_values[i, j]
                if Ti >= Ti_min:
                    theta = Ti / (1 - Ti * (C2 + Ti * (C4 + Ti * C6)) / (1 + Ti * (C3 + Ti * (C5 + Ti * C7))))
                    xi = (B_G**2 / (4*theta))**(1/3)
                    sigma_v = C1 * theta * np.sqrt(xi / (mrcc * Ti**3)) * np.exp(-3*xi)
                else:
                    sigma_v = 0
                sigma_v_values[i, j] = sigma_v

            return ni_values**2 * sigma_v_values

        # Integrate to find the total number of neutrons
        # Units of s^-1
        # Need to check if integral is correct
        integrand = lambda z, r: get_reaction_rate(r, z) * r
        integral = integrate.simps(integrate.simps(integrand(z, r), z), r)
        self.total_neutrons = integral * 2 * np.pi
        
        neutron_r_profile = get_reaction_rate(r, 0)[:, 0]
        neutron_z_profile = get_reaction_rate((plasma_outerRadius + plasma_innerRadius)/2, z)[0, :]

        return r, z, neutron_r_profile, neutron_z_profile

    def set_source(self):
        # SOURCE
        # Create a CMFX-like neutron source
        self.source = openmc.Source(domains=[self.plasma_cell])

        r, z, neutron_r_profile, neutron_z_profile = self.get_neutron_profile()

        r_dist = openmc.stats.Tabular(r, neutron_r_profile)
        angle_dist = openmc.stats.Uniform(a=0.0, b=2*np.pi)
        z_dist = openmc.stats.Tabular(z, neutron_z_profile)

        # Normalize r and z profiles so the integral == 1
        r_dist.normalize()
        z_dist.normalize()

        self.source.space = openmc.stats.CylindricalIndependent(r=r_dist, phi=angle_dist, z=z_dist, origin=(0.0, 0.0, 0.0))
        self.source.angle = openmc.stats.Isotropic()
        if self.fuel == 'DD':
            self.source.energy = openmc.stats.muir(e0=2.45e6, m_rat=4.0, kt=self.Ti_peak)
        elif self.fuel == 'DT':
            self.source.energy = openmc.stats.muir(e0=14.1e6, m_rat=5.0, kt=self.Ti_peak)
        # Sets the source strength to the total neutron production rate in n/s
        self.source.strength = self.total_neutrons

    def run_settings(self):
        self.settings = openmc.Settings()
        self.settings.batches = 10
        self.settings.particles = self.particles
        self.settings.run_mode = "fixed source"
        self.settings.source = self.source


    def create_mesh(self, N_points=101):
        mesh = openmc.RegularMesh()
        mesh.dimension = [N_points, N_points, N_points]
        mesh.lower_left = [-plasma_outerRadius, -plasma_outerRadius, -plasma_length / 2]
        mesh.upper_right = [plasma_outerRadius, plasma_outerRadius, plasma_length / 2]
        self.N_points = N_points

        return mesh

    def set_tallies(self):
        self.tallies = openmc.Tallies()

        cell_tally = openmc.Tally(name="tally_in_cell")
        cell_filter = openmc.CellFilter(self.He3_cell)
        cell_tally.scores = ["absorption"]
        cell_tally.filters = [cell_filter]
        self.tallies.append(cell_tally)

        if self.create_mesh_tally:
            # Create mesh tally to score flux
            mesh_tally = openmc.Tally(name='tallies_on_mesh')
            # Create mesh filter for tally
            mesh = self.create_mesh()
            mesh_filter = openmc.MeshFilter(mesh)
            mesh_tally.filters = [mesh_filter]
            mesh_tally.scores = ['flux']
            self.tallies.append(mesh_tally)

    def run(self, directory):
        if self.neutrons_generated:
            self.directory = directory
            self.model = openmc.model.Model(self.geometry, self.materials, self.settings, self.tallies)
            try:
                self.sp_filename = self.model.run(cwd=self.directory, threads=n_threads)
            except RuntimeError:
                print('Too many particles lost')

            # Save the source object to pickle
            with open(f'{self.directory}/{source_pkl}', 'wb') as file:
                pickle.dump(self, file)
        else:
            print('The plasma was too cold to generate any neutrons')

    def read_results(self):
        # open the results file
        self.sp = openmc.StatePoint(self.sp_filename)

        # access the tally using pandas dataframes
        self.tally = self.sp.get_tally(name='tally_in_cell')
        self.results = self.tally.get_pandas_dataframe()

        return self.results
    
    def plot_flux(self):
        if self.create_mesh_tally:
            if not hasattr(self, 'sp'):
                self.sp = openmc.StatePoint(self.sp_filename)
            
            flux_tally = self.sp.get_tally(name='tallies_on_mesh')
            flux_df = flux_tally.get_pandas_dataframe()

            # The dataframe only has the indices for the mesh geometry, not their actual values
            # We will replace the indices with values
            x, dx = np.linspace(-plasma_outerRadius, plasma_outerRadius, self.N_points, retstep=True)
            y, dy = np.linspace(-plasma_outerRadius, plasma_outerRadius, self.N_points, retstep=True)
            z, dz = np.linspace(-plasma_length / 2, plasma_length / 2, self.N_points, retstep=True)

            # Normalize to particle / cm^2-s because flux comes in particle-cm
            # By setting the source strength, all results are totals, not per source particle
            mesh_cell_volume = dx * dy * dz
            flux_df['mean'] = flux_df['mean'] / mesh_cell_volume
            flux_df['std. dev.'] = flux_df['std. dev.'] / mesh_cell_volume

            # Plot the figures such that the y axis height is the same
            # Note that the ratio is hardcoded in for the specific height and width of the current plasma, look for better solution in future
            fig = plt.figure(figsize=(12, 4))
            aspect_ratio = np.ptp(z) / np.ptp(x)
            gs = fig.add_gridspec(1, 2,  width_ratios=(1, aspect_ratio*1.035),
                      left=0.1, right=0.9, bottom=0.1, top=0.9,
                      wspace=0.05, hspace=0.05)
            ax1 = fig.add_subplot(gs[0])
            ax2 = fig.add_subplot(gs[1], sharey=ax1)

            # The aspect ratio is equal
            ax1.set_aspect('equal', adjustable='box')
            ax2.set_aspect('equal', adjustable='box')

            # Perpendicular slice
            slice_df = flux_df[flux_df[('mesh 1', 'z')] == int((self.N_points - 1) / 2)]
            [X, Y] = np.meshgrid(x, y)
            values = slice_df['mean'].to_numpy().reshape((self.N_points, self.N_points))
            im = ax1.contourf(X, Y, values, levels=15)
            ax1.set_xlabel('x (cm)')

            # Parallel slice
            slice_df = flux_df[flux_df[('mesh 1', 'x')] == int((self.N_points - 1) / 2)]
            [Z, Y] = np.meshgrid(z, y)
            values = slice_df['mean'].to_numpy().reshape((self.N_points, self.N_points)).T
            im = ax2.contourf(Z, Y, values, levels=15)
            ax2.set_xlabel('z (cm)')
            # Remove y tick labels
            ax2.tick_params(axis='y', which='both', labelleft=False)

            # Make colorbar same height as plots
            divider = make_axes_locatable(ax2)
            cax = divider.append_axes("right", size="2%", pad=0.1)
            fig.colorbar(im, cax=cax, label=r'Flux $\left( \frac{\#}{ \mathrm{cm}^2 \mathrm{s} } \right)$')
            ax1.set_ylabel('y (cm)')

            # fig.set_constrained_layout(False)

            plt.savefig(f'{self.directory}/flux.png', dpi=200)
            plt.show()
        else:
            print('Did not create mesh tally, so not plotting flux')