import openmc
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import neutronics_material_maker as nmm
from constants import *
import pickle

class Point_Source():
    def __init__(self, activity=1, shielding=False, particles=250000, distance=20, moderator=True, create_mesh_tally=False):
        self.activity = activity
        self.shielding = shielding
        self.particles = particles
        self.distance = distance
        self.moderator = moderator
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

        self.HDPE_material = nmm.Material.from_library(name='Polyethylene, Non-borated').openmc_material
        self.HDPE_material.set_density("g/cm3", 0.95) # Need to change density to HDPE instead of LDPE
        self.HDPE_material.add_s_alpha_beta('c_H_in_CH2') # Using thermal scattering data, need to compare to https://www.sciencedirect.com/science/article/pii/0168900287903706
        self.Pb_material = nmm.Material.from_library(name='Lead').openmc_material

        self.materials = openmc.Materials([self.He3_material, self.Al_material, self.HDPE_material, self.Pb_material])

    def set_geometry(self):
        # SURFACES
        void_surface = openmc.Sphere(r=void_radius, boundary_type='vacuum')
        He3_cylinder = openmc.model.RightCircularCylinder((self.distance, -He3_length / 2, 0),
                                                          He3_length, He3_diameter / 2, axis='y')
        detector_cylinder = openmc.model.RightCircularCylinder((self.distance, -He3_length / 2 - gas_offset, 0),
                                                               detector_length, detector_diameter / 2, axis='y')
        HDPE_surface = openmc.model.RightCircularCylinder((self.distance, -HDPE_height / 2, 0),
                                                          HDPE_height, HDPE_diameter / 2, axis='y')
        Pb_surface = openmc.model.RightCircularCylinder((self.distance, -HDPE_height / 2 - Pb_thickness, 0),
                                                          HDPE_height + 2*Pb_thickness, HDPE_diameter / 2 + Pb_thickness, axis='y')
        
        # Aluminum enclosure
        xmin = self.distance - enclosure_length / 2
        xmax = self.distance + enclosure_length / 2
        ymin = -HDPE_height / 2 - Pb_thickness
        ymax = -HDPE_height / 2 - Pb_thickness + enclosure_height
        zmin = -enclosure_width / 2
        zmax = enclosure_width / 2
        enclosure_innerSurface = openmc.model.RectangularParallelepiped(xmin, xmax, ymin, ymax, zmin, zmax)
        enclosure_outerSurface = openmc.model.RectangularParallelepiped(xmin - enclosure_thickness, xmax + enclosure_thickness,
                                                                        ymin - enclosure_thickness, ymax + enclosure_thickness,
                                                                        zmin - enclosure_thickness, zmax + enclosure_thickness)

        # CELLS
        He3_region = -He3_cylinder
        detector_region = (-detector_cylinder & +He3_cylinder)
        HDPE_region = (-HDPE_surface & +detector_cylinder)
        Pb_region = (-Pb_surface & +HDPE_surface)
        enclosure_region = (-enclosure_outerSurface & +enclosure_innerSurface)
        if not self.moderator:
            void_region = -void_surface & +detector_cylinder
        elif not self.shielding:
            void_region = -void_surface & +HDPE_surface
        else:
            void_region = -void_surface & +Pb_surface & ~enclosure_region

        self.void_cell = openmc.Cell(region=void_region)
        self.He3_cell = openmc.Cell(region=He3_region)
        self.detector_cell = openmc.Cell(region=detector_region)
        self.HDPE_cell = openmc.Cell(region=HDPE_region)
        self.Pb_cell = openmc.Cell(region=Pb_region)
        self.enclosure_cell = openmc.Cell(region=enclosure_region)

        self.He3_cell.fill = self.He3_material
        self.detector_cell.fill = self.Al_material
        self.HDPE_cell.fill = self.HDPE_material
        self.Pb_cell.fill = self.Pb_material
        self.enclosure_cell.fill = self.Al_material

        self.universe = openmc.Universe(cells=[self.void_cell, self.He3_cell, self.detector_cell])
        if self.moderator:
            self.universe.add_cells([self.HDPE_cell])
        if self.shielding:
            self.universe.add_cells([self.Pb_cell, self.enclosure_cell])
        self.geometry = openmc.Geometry(root=self.universe)

    def set_source(self):
        # SOURCE
        # Create a neutron point source
        self.source = openmc.Source()
        self.source.space = openmc.stats.Point((0, 0, 0))

        # radius = openmc.stats.Uniform(0, 0.2)
        # angle = openmc.stats.Uniform(0.0, 2*np.pi)
        # z_values = openmc.stats.Normal(0, 1)

        # self.source.space = openmc.stats.CylindricalIndependent(r=radius, phi=angle, z=z_values, origin=(0.0, 0.0, 0.0))
        self.source.angle = openmc.stats.Isotropic()

        # Distribution for Cf252
        self.source.energy = openmc.stats.Watt(a=1.18e6, b=1.03419e-6)

        # Based on activity of source
        self.source.strength = self.activity

    def run_settings(self):
        self.settings = openmc.Settings()
        self.settings.batches = 100
        self.settings.particles = self.particles
        self.settings.run_mode = "fixed source"
        self.settings.source = self.source

    def create_mesh(self, N_points=101):
        mesh = openmc.RegularMesh.from_domain(self.He3_cell, dimension=(N_points, N_points, N_points), mesh_id=0)
        self.N_points = N_points

        return mesh

    def set_tallies(self):
        self.tallies = openmc.Tallies()

        cell_tally = openmc.Tally(name='tally_in_cell')
        cell_filter = openmc.CellFilter(self.He3_cell)
        cell_tally.scores = ['absorption', 'flux']
        cell_tally.filters = [cell_filter]
        self.tallies.append(cell_tally)

        if self.create_mesh_tally:
            # Create mesh tally to score flux
            mesh_tally = openmc.Tally(name='tallies_on_mesh')
            # Create mesh filter for tally
            mesh = self.create_mesh()
            mesh_filter = openmc.MeshFilter(mesh)
            mesh_tally.filters = [mesh_filter, cell_filter]
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
    
    def get_flux_map(self):
        if self.create_mesh_tally:
            if not hasattr(self, 'sp'):
                self.sp = openmc.StatePoint(self.sp_filename)
            
            flux_tally = self.sp.get_tally(name='tallies_on_mesh')
            flux_df = flux_tally.get_pandas_dataframe()

            # The dataframe only has the indices for the mesh geometry, not their actual values
            # We will replace the indices with values
            x, dx = np.linspace(-He3_diameter / 2, He3_diameter / 2, self.N_points, retstep=True)
            y, dy = np.linspace(-He3_length / 2, He3_length / 2, self.N_points, retstep=True)
            z, dz = np.linspace(-He3_diameter / 2, He3_diameter / 2, self.N_points, retstep=True)

            # Normalize to particle / cm^2-s because flux comes in particle-cm
            # By setting the source strength, all results are totals, not per source particle
            mesh_cell_volume = dx * dy * dz
            flux_df['mean'] = flux_df['mean'] / mesh_cell_volume
            flux_df['std. dev.'] = flux_df['std. dev.'] / mesh_cell_volume

            # Plot the figures such that the y axis height is the same
            # Note that the ratio is hardcoded in for the specific height and width of the current plasma, look for better solution in future
            # fig = plt.figure(figsize=(12, 4))
            # aspect_ratio = np.ptp(z) / np.ptp(x)
            # gs = fig.add_gridspec(1, 2,  width_ratios=(1, aspect_ratio*1.035),
            #           left=0.1, right=0.9, bottom=0.1, top=0.9,
            #           wspace=0.05, hspace=0.05)
            # ax1 = fig.add_subplot(gs[0])
            # ax2 = fig.add_subplot(gs[1], sharey=ax1)

            # # The aspect ratio is equal
            # ax1.set_aspect('equal', adjustable='box')
            # ax2.set_aspect('equal', adjustable='box')

            # Perpendicular slice
            # We set the mesh_id to 0
            slice_df = flux_df[flux_df[(f'mesh 0', 'x')] == int((self.N_points - 1) / 2)]
            [Z, Y] = np.meshgrid(z, y)
            values = slice_df['mean'].to_numpy().reshape((self.N_points, self.N_points)).T

            return Z, Y, values
            # im = ax1.contourf(X, Y, values, levels=15)
            # ax1.set_xlabel('x (cm)')

            # # Parallel slice
            # slice_df = flux_df[flux_df[('mesh 1', 'x')] == int((self.N_points - 1) / 2)]
            # [Z, Y] = np.meshgrid(z, y)
            # values = slice_df['mean'].to_numpy().reshape((self.N_points, self.N_points)).T
            # im = ax2.contourf(Z, Y, values, levels=15)
            # ax2.set_xlabel('z (cm)')
            # # Remove y tick labels
            # ax2.tick_params(axis='y', which='both', labelleft=False)

            # # Make colorbar same height as plots
            # divider = make_axes_locatable(ax2)
            # cax = divider.append_axes("right", size="2%", pad=0.1)
            # fig.colorbar(im, cax=cax, label=r'Flux $\left( \frac{\#}{ \mathrm{cm}^2 \mathrm{s} } \right)$')
            # ax1.set_ylabel('y (cm)')

            # # fig.set_constrained_layout(False)

            # plt.savefig(f'{self.directory}/flux.png', dpi=200)
            # plt.show()
        else:
            print('Did not create mesh tally, so not plotting flux')


