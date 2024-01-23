from constants import *
import openmc
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import neutronics_material_maker as nmm
import pickle
from scipy import integrate, interpolate

TORR2DENSITY = 1.622032e-7 # At 25 C
IN2CM = 2.54

class CMFX_Source():
    def __init__(self, particles=250000, radialDistance=50, axialDistance=0, create_mesh_tally=False):
        self.particles = particles
        self.radialDistance = radialDistance
        self.axialDistance = axialDistance
        self.create_mesh_tally = create_mesh_tally

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
        self.HDPE_material.set_density("g/cm3", 0.95) # Need to change density to HDPE instead of LDPE
        self.HDPE_material.add_s_alpha_beta('c_H_in_CH2') # Using thermal scattering data, need to compare to https://www.sciencedirect.com/science/article/pii/0168900287903706
        self.Pb_material = nmm.Material.from_library(name='Lead').openmc_material

        self.plasma_material = nmm.Material.from_library(name='DD_plasma').openmc_material

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
        
        # Add volume calculation to plasma so we can normalize flux later
        self.plasma_cell.volume = np.pi * plasma_length * (plasma_outerRadius**2 - plasma_innerRadius**2)

        self.universe = openmc.Universe(cells=[self.void_cell, self.He3_cell, self.detector_cell,
                                               self.HDPE_cell, self.Pb_cell, self.enclosure_cell,
                                               self.centerConductor_cell, self.chamber_cell, self.plasma_cell])
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
        Ti_profile = T_peak * r_profile * z_profile
        ni_profile = n_peak * r_profile * z_profile

        # Create interpolation functions so we can pass them to the integral later
        f_Ti = interpolate.RectBivariateSpline(r, z, Ti_profile)
        f_ni = interpolate.RectBivariateSpline(r, z, ni_profile)

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

        # Eqns 12-14 of above paper
        # Returns <sigma * v> in cm^3 / s
        # We also multiply by density^2 to get units of cm^-3 s^-1
        # Need this to be a function of r and z so that we can integrate over it and get the total neutrons
        def get_reaction_rate(r, z):
            Ti = f_Ti(r, z)
            ni = f_ni(r, z)

            theta = Ti / (1 - Ti * (C2 + Ti * (C4 + Ti * C6)) / (1 + Ti * (C3 + Ti * (C5 + Ti * C7))))
            xi = (B_G**2 / (4*theta))**(1/3)
            sigma_v = C1 * theta * np.sqrt(xi / (mrcc * Ti**3)) * np.exp(-3*xi)

            # Remove nan's that exist because we divide by 0
            sigma_v = np.nan_to_num(sigma_v)

            return ni**2 * sigma_v

        # Integrate to find the total number of neutrons
        # Units of s^-1
        integrand = lambda z, r: get_reaction_rate(r, z) * r
        integral, error = integrate.dblquad(integrand, plasma_innerRadius, plasma_outerRadius,
                                                            -plasma_length / 2, plasma_length / 2)
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
        self.source.energy = openmc.stats.muir(e0=2.45e6, m_rat=4.0, kt=10000)
        self.source.strength = self.total_neutrons * duration

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
        self.directory = directory
        self.model = openmc.model.Model(self.geometry, self.materials, self.settings, self.tallies)
        try:
            self.sp_filename = self.model.run(cwd=self.directory, threads=n_threads)
        except RuntimeError:
            print('Too many particles lost')

        # Save the source object to pickle
        with open(f'{self.directory}/{source_pkl}', 'wb') as file:
            pickle.dump(self, file)

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
            mesh_cell_volume = dx * dy * dz
            flux_df['mean'] = flux_df['mean'] / (mesh_cell_volume * duration)
            flux_df['std. dev.'] = flux_df['std. dev.'] / (mesh_cell_volume * duration)

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

            fig.set_constrained_layout(False)

            plt.savefig(f'{self.directory}/flux.png', dpi=200)
            plt.show()
        else:
            print('Did not create mesh tally, so not plotting flux')