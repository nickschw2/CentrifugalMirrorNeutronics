from constants import *
from magnetic_field import *
import openmc
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import neutronics_material_maker as nmm
import pickle
from tqdm import tqdm
from scipy import integrate, interpolate

class CMFX_Source():
    def __init__(self, Ti_avg=1, ni_avg=1e13, particles=250000, radialDistance=50, axialDistance=0,
                 fuel='DD', profile='parabolic', create_mesh_tally=False):
        self.Ti_avg = Ti_avg
        self.ni_avg = ni_avg
        self.particles = particles
        self.radialDistance = radialDistance
        self.axialDistance = axialDistance
        self.fuel = fuel
        self.profile = profile
        self.create_mesh_tally = create_mesh_tally

        # Determine whether the plasma is too cold to generate any neutrons, which would lead to runtime error in simulation
        self.neutrons_generated = self.Ti_avg >= Ti_min

        self.set_materials()
        self.set_geometry()
        self.set_source()
        self.run_settings()
        self.set_tallies()

    def set_materials(self):
        # There is no thermal scattering data for He3
        self.He3_material = nmm.Material.from_library(name='He-3 proportional gas').openmc_material
        self.He3_material.set_density("g/cm3", gas_pressure * TORR2DENSITY)

        self.Al_material = nmm.Material.from_library(name='Aluminum, alloy 6061-O').openmc_material
        self.Steel_material = nmm.Material.from_library(name='Steel, Stainless 316L').openmc_material

        self.HDPE_material = nmm.Material.from_library(name='Polyethylene, Non-borated').openmc_material
        self.HDPE_material.set_density("g/cm3", 0.95)
        self.HDPE_material.add_s_alpha_beta('c_H_in_CH2') # Using thermal scattering data, need to compare to https://www.sciencedirect.com/science/article/pii/0168900287903706
        self.Pb_material = nmm.Material.from_library(name='Lead').openmc_material

        if self.fuel == 'DD':
            self.plasma_material = nmm.Material.from_library(name='DD_plasma').openmc_material
            self.plasma_material.set_density("g/cm3", self.ni_avg * 2 * ProtonMass)
        elif self.fuel == 'DT':
            self.plasma_material = nmm.Material.from_library(name='DT_plasma').openmc_material
            self.plasma_material.set_density("g/cm3", self.ni_avg * 2.5 * ProtonMass)

        self.materials = openmc.Materials([self.He3_material, self.Al_material, self.Steel_material,
                                           self.HDPE_material, self.Pb_material, self.plasma_material])

    def set_geometry(self):
        ### SURFACES ###
        # Detector Surfaces
        void_surface = openmc.model.RightCircularCylinder((0, 0, -void_length / 2), void_length, void_radius,
                                                          axis='z', boundary_type='vacuum')
        He3_cylinder = openmc.model.RightCircularCylinder((self.radialDistance, -He3_length / 2, self.axialDistance),
                                                          He3_length, He3_diameter / 2, axis='y')
        detector_cylinder = openmc.model.RightCircularCylinder((self.radialDistance, -He3_length / 2 - gas_offset, self.axialDistance),
                                                               detector_length, detector_diameter / 2, axis='y')
        HDPE_surface = openmc.model.RightCircularCylinder((self.radialDistance, -HDPE_height / 2, self.axialDistance),
                                                          HDPE_height, HDPE_diameter / 2, axis='y')
        Pb_surface = openmc.model.RightCircularCylinder((self.radialDistance, -HDPE_height / 2 - Pb_thickness, self.axialDistance),
                                                          HDPE_height + 2*Pb_thickness, HDPE_diameter / 2 + Pb_thickness, axis='y')
        
        # Aluminum enclosure
        xmin = self.radialDistance - enclosure_length / 2
        xmax = self.radialDistance + enclosure_length / 2
        ymin = -HDPE_height / 2 - Pb_thickness
        ymax = -HDPE_height / 2 - Pb_thickness + enclosure_height
        zmin = self.axialDistance - enclosure_width / 2
        zmax = self.axialDistance + enclosure_width / 2
        enclosure_innerSurface = openmc.model.RectangularParallelepiped(xmin, xmax, ymin, ymax, zmin, zmax)
        enclosure_outerSurface = openmc.model.RectangularParallelepiped(xmin - enclosure_thickness, xmax + enclosure_thickness,
                                                                        ymin - enclosure_thickness, ymax + enclosure_thickness,
                                                                        zmin - enclosure_thickness, zmax + enclosure_thickness)

        # Plasma Surfaces
        plasma_outerSurface = openmc.model.RightCircularCylinder((0, 0, -plasma_length / 2), plasma_length,
                                                                 plasma_outerRadius, axis='z')
        # The plasma doesn't go all the way to the center conductor radius, but we say it does to capture
        # all the mesh points in the surfaces bound by plasma_region
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

        # CELLS
        He3_region = -He3_cylinder
        detector_region = (-detector_cylinder & +He3_cylinder)
        HDPE_region = (-HDPE_surface & +detector_cylinder)
        Pb_region = (-Pb_surface & +HDPE_surface)
        enclosure_region = (-enclosure_outerSurface & +enclosure_innerSurface)
        plasma_region = (-plasma_outerSurface & +plasma_innerSurface)
        centerConductor_region = (-centerConductor_outerSurface & +centerConductor_innerSurface)
        chamber_region = (-chamber_outerSurface & +chamber_innerSurface)
        void_region = -void_surface & +Pb_surface & ~enclosure_region & ~plasma_region & ~centerConductor_region & ~chamber_region

        self.void_cell = openmc.Cell(region=void_region)
        self.He3_cell = openmc.Cell(region=He3_region)
        self.detector_cell = openmc.Cell(region=detector_region)
        self.HDPE_cell = openmc.Cell(region=HDPE_region)
        self.Pb_cell = openmc.Cell(region=Pb_region)
        self.enclosure_cell = openmc.Cell(region=enclosure_region)
        self.plasma_cell = openmc.Cell(region=plasma_region)
        self.centerConductor_cell = openmc.Cell(region=centerConductor_region)
        self.chamber_cell = openmc.Cell(region=chamber_region)

        self.He3_cell.fill = self.He3_material
        self.detector_cell.fill = self.Al_material
        self.HDPE_cell.fill = self.HDPE_material
        self.Pb_cell.fill = self.Pb_material
        self.enclosure_cell.fill = self.Al_material
        self.plasma_cell.fill = self.plasma_material
        self.centerConductor_cell.fill = self.Steel_material
        self.chamber_cell.fill = self.Al_material

        self.universe = openmc.Universe(cells=[self.void_cell, self.He3_cell, self.detector_cell,
                                               self.HDPE_cell, self.Pb_cell, self.enclosure_cell,
                                               self.centerConductor_cell, self.chamber_cell, self.plasma_cell])
        self.geometry = openmc.Geometry(root=self.universe)

    def get_neutron_profile(self):
        R, Z, Ti_profile, ni_profile = get_profiles(self.Ti_avg, self.ni_avg, profile=self.profile)

        # Create interpolation functions so we can pass them to the integral later
        f_Ti = interpolate.RectBivariateSpline(R, Z, Ti_profile.T)
        f_ni = interpolate.RectBivariateSpline(R, Z, ni_profile.T)
        
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
            delta_ij = 1
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
            delta_ij = 0

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

            return ni_values**2 / (1 + delta_ij) * sigma_v_values
        
        # Need a function to normalize the strengths from n/cm^3/s to n/s
        # This way we can sum over all the strengths and have them sum to the total strength
        def normalized_strength(r, z):
            strengths = get_reaction_rate(R, Z) # n/cm^3/s
            RR, ZZ = np.meshgrid(r, z, indexing='ij')
            volumes = np.diff(r**2, prepend=[(2*r[0] - r[1])**2])[:, np.newaxis] * np.abs(np.diff(z, prepend=[2*z[0] - z[1]]))[np.newaxis, :] * np.pi
            strengths_normalized = strengths * volumes # n/s
            return strengths_normalized

        # Integrate to find the total number of neutrons
        # Units of s^-1
        # Need to check if integral is correct
        integrand = lambda r, z: get_reaction_rate(r, z).T * r
        integral = integrate.simpson(integrate.simpson(integrand(R, Z), x=R), x=Z)
        self.total_neutrons = integral * 2 * np.pi
        print(self.total_neutrons)
        strengths = normalized_strength(R, Z)
        # Check that we've normalized correctly to within 1%
        if (self.total_neutrons - np.sum(strengths)) / self.total_neutrons > 0.01:
            print('Check for correct normalization of strengths')

        # Plot original strengths
        if self.create_mesh_tally:
            fig, ax = plt.subplots(layout='constrained')
            cs = ax.contourf(Z, R, strengths, np.linspace(1e-6, np.max(strengths), 100))
            fig.colorbar(cs, ax=ax, label='n cm$^{-3}$ s$^{-1}$', orientation='horizontal', ticks=np.linspace(0, np.max(strengths), 8).astype('int'))
            ax.set_xlabel('z (cm)')
            ax.set_ylabel('r (cm)')
            ax.set_aspect('equal')
            ax.set_ylim(min(R), max(R))
            plt.savefig(f'{figures_folder}/source_strength_{self.profile}.png', bbox_inches='tight', dpi=300)
            plt.close()
    
        return R, Z, strengths

    def set_source(self):
        # SOURCE
        # Create a CMFX-like neutron source
        r, z, strengths = self.get_neutron_profile()
        cylindrical_mesh = openmc.CylindricalMesh(r_grid=r, z_grid=z, phi_grid=[0, 2*np.pi])

        # Need to remove last row and column when using spatial mesh because the vertices are 1 longer than the mesh cells
        # Need to transpose as well because openmc uses np.flatten(strengths) in the wrong direction
        strengths = strengths[:-1, :-1].T

        self.source = openmc.IndependentSource()
        self.source.space = openmc.stats.MeshSpatial(cylindrical_mesh, strengths=strengths, volume_normalized=False)
        self.source.angle = openmc.stats.Isotropic()
        if self.fuel == 'DD':
            self.source.energy = openmc.stats.muir(e0=2.45e6, m_rat=4.0, kt=self.Ti_avg)
        elif self.fuel == 'DT':
            self.source.energy = openmc.stats.muir(e0=14.1e6, m_rat=5.0, kt=self.Ti_avg)
        self.source.strength = self.total_neutrons

    def run_settings(self):
        self.settings = openmc.Settings()
        self.settings.batches = batches
        self.settings.particles = self.particles
        self.settings.run_mode = "fixed source"
        self.settings.source = self.source


    def create_mesh(self, N_points=101):
        mesh = openmc.RegularMesh()
        mesh.dimension = [N_points, N_points, N_points]
        mesh.lower_left = [-plasma_outerRadius, -plasma_outerRadius, -plasma_length / 2]
        mesh.upper_right = [plasma_outerRadius, plasma_outerRadius, plasma_length / 2]

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
            self.mesh = self.create_mesh()
            mesh_filter = openmc.MeshFilter(self.mesh)
            time_filter = openmc.TimeFilter([0, 1e-12])
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
            # with open(f'{self.directory}/{source_pkl}', 'wb') as file:
            #     pickle.dump(self, file)
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
            nx, ny, nz = self.mesh.dimension
            x, dx = np.linspace(self.mesh.lower_left[0], self.mesh.upper_right[0], nx, retstep=True)
            y, dy = np.linspace(self.mesh.lower_left[1], self.mesh.upper_right[1], ny, retstep=True)
            z, dz = np.linspace(self.mesh.lower_left[2], self.mesh.upper_right[2], nz, retstep=True)

            # Normalize to particle / cm^2-s because flux comes in particle-cm
            # By setting the source strength, all results are totals, not per source particle
            mesh_cell_volume = dx * dy * dz
            flux_df['mean'] = flux_df['mean'] / mesh_cell_volume
            flux_df['std. dev.'] = flux_df['std. dev.'] / mesh_cell_volume

            # Plot the figures such that the y axis height is the same
            # Note that the ratio is hardcoded in for the specific height and width of the current plasma, look for better solution in future
            fig = plt.figure(figsize=(12, 4))
            aspect_ratio = np.ptp(z) / np.ptp(x)
            gs = fig.add_gridspec(1, 2,  width_ratios=(1, aspect_ratio*1.035), left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.05, hspace=0.05)
            ax1 = fig.add_subplot(gs[0])
            ax2 = fig.add_subplot(gs[1], sharey=ax1)

            # The aspect ratio is equal
            ax1.set_aspect('equal', adjustable='box')
            ax2.set_aspect('equal', adjustable='box')

            # Perpendicular slice
            slice_df = flux_df[flux_df[('mesh 2', 'z')] == int((nz - 1) / 2)]
            [X, Y] = np.meshgrid(x, y)
            values = slice_df['mean'].to_numpy().reshape((len(y), len(x)))
            im = ax1.contourf(X, Y, values, levels=15)
            ax1.set_xlabel('x (cm)')

            # Parallel slice
            slice_df = flux_df[flux_df[('mesh 2', 'x')] == int((nx-1) / 2)]
            [Y, Z] = np.meshgrid(y, z)
            values = slice_df['mean'].to_numpy().reshape((len(y), len(z)))
            im = ax2.contourf(Z, Y, values, levels=15)
            ax2.set_xlabel('z (cm)')
            # Remove y tick labels
            ax2.tick_params(axis='y', which='both', labelleft=False)

            # Make colorbar same height as plots
            divider = make_axes_locatable(ax2)
            cax = divider.append_axes("right", size="2%", pad=0.1)
            fig.colorbar(im, cax=cax, label=r'Flux $\left( \frac{\#}{ \mathrm{cm}^2 \mathrm{s} } \right)$')
            ax1.set_ylabel('y (cm)')

            fig.set_constrained_layout(False)

            plt.savefig(f'{figures_folder}/neutron_flux_{self.profile}.png', dpi=300, bbox_inches='tight')
            plt.show()
        else:
            print('Did not create mesh tally, so not plotting flux')