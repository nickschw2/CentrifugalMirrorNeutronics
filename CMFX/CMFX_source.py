from constants import *
import openmc
import numpy as np
import matplotlib.pyplot as plt
import neutronics_material_maker as nmm
import pickle

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

        self.detector_material = nmm.Material.from_library(name='Aluminum, alloy 6061-O').openmc_material

        self.HDPE_material = nmm.Material.from_library(name='Polyethylene, Non-borated').openmc_material
        self.HDPE_material.set_density("g/cm3", 0.95) # Need to change density to HDPE instead of LDPE
        self.HDPE_material.add_s_alpha_beta('c_H_in_CH2') # Using thermal scattering data, need to compare to https://www.sciencedirect.com/science/article/pii/0168900287903706
        self.Pb_material = nmm.Material.from_library(name='Lead').openmc_material

        self.hydrogen_material = nmm.Material.from_library(name='DD_plasma').openmc_material

        self.materials = openmc.Materials([self.He3_material, self.detector_material, self.HDPE_material, self.Pb_material, self.hydrogen_material])

    def set_geometry(self):
        # SURFACES
        # large radius as MeV neutrons travel at cm per nano second (1e-9s)
        void_surface = openmc.model.RightCircularCylinder((0, 0, -void_length / 2), void_length, void_radius,
                                                          axis='z', boundary_type='vacuum')
        He3_cylinder = openmc.model.RightCircularCylinder((self.radialDistance, -He3_length / 2, self.axialDistance),
                                                          He3_length, He3_diameter / 2, axis='y')
        detector_cylinder = openmc.model.RightCircularCylinder((self.radialDistance, -detector_length / 2, self.axialDistance),
                                                               detector_length, detector_diameter / 2, axis='y')
        HDPE_surface = openmc.model.RightCircularCylinder((self.radialDistance, -HDPE_height / 2, self.axialDistance),
                                                          HDPE_height, HDPE_diameter / 2, axis='y')
        Pb_surface = openmc.model.RightCircularCylinder((self.radialDistance, -HDPE_height / 2 + Pb_thickness, self.axialDistance),
                                                          HDPE_height + 2*Pb_thickness, HDPE_diameter / 2 + Pb_thickness, axis='y')

        plasma_outerSurface = openmc.model.RightCircularCylinder((0, 0, -plasma_length / 2), plasma_length,
                                                          plasma_outerRadius, axis='z')
        plasma_innerSurface = openmc.model.RightCircularCylinder((0, 0, -plasma_length / 2), plasma_length,
                                                          plasma_innerRadius, axis='z')

        # CELLS
        He3_region = -He3_cylinder
        detector_region = (-detector_cylinder & +He3_cylinder)
        HDPE_region = (-HDPE_surface & +detector_cylinder)
        Pb_region = (-Pb_surface & +HDPE_surface)
        plasma_region = (-plasma_outerSurface & +plasma_innerSurface)
        void_region = -void_surface & +HDPE_surface & ~plasma_region

        self.void_cell = openmc.Cell(region=void_region)
        self.He3_cell = openmc.Cell(region=He3_region)
        self.detector_cell = openmc.Cell(region=detector_region)
        self.HDPE_cell = openmc.Cell(region=HDPE_region)
        self.Pb_cell = openmc.Cell(region=Pb_region)
        self.plasma_cell = openmc.Cell(region=plasma_region)

        self.He3_cell.fill = self.He3_material
        self.detector_cell.fill = self.detector_material
        self.HDPE_cell.fill = self.HDPE_material
        self.Pb_cell.fill = self.HDPE_material
        self.plasma_cell.fill = self.hydrogen_material

        self.universe = openmc.Universe(cells=[self.void_cell, self.He3_cell, self.detector_cell, self.HDPE_cell, self.Pb_cell, self.plasma_cell])
        self.geometry = openmc.Geometry(root=self.universe)

    def set_source(self):
        # SOURCE
        # Create a CMFX-like neutron source
        self.source = openmc.Source(domains=[self.plasma_cell])

        radius = openmc.stats.Normal(plasma_innerRadius + plasma_outerRadius / 2, (plasma_outerRadius - plasma_innerRadius) / 4)
        angle = openmc.stats.Uniform(a=0.0, b=2*np.pi)
        z_values = openmc.stats.Normal(0, plasma_length / 4)

        self.source.space = openmc.stats.CylindricalIndependent(r=radius, phi=angle, z=z_values, origin=(0.0, 0.0, 0.0))
        self.source.angle = openmc.stats.Isotropic()
        self.source.energy = openmc.stats.muir(e0=2.45e6, m_rat=4.0, kt=10000)
        

    def run_settings(self):
        self.settings = openmc.Settings()
        self.settings.batches = 100
        self.settings.particles = self.particles
        self.settings.run_mode = "fixed source"
        self.settings.source = self.source

    def create_mesh(self, N_points=101):
        mesh = openmc.RegularMesh()
        mesh.dimension = [N_points, N_points, N_points]
        mesh.lower_left = [-plasma_outerRadius, -plasma_outerRadius, -plasma_length]
        mesh.upper_right = [plasma_outerRadius, plasma_outerRadius, plasma_length]
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
            self.sp_filename = self.model.run(cwd=self.directory, threads=4)
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

        # Normalize to total counts
        self.results['mean'] = self.results['mean'] * self.particles
        self.results['std. dev.'] = self.results['std. dev.'] * self.particles

        return self.results
    
    def plot_flux(self, section='perp'):
        if self.create_mesh_tally:
            if not hasattr(self, 'sp'):
                self.sp = openmc.StatePoint(self.sp_filename)
            
            flux_tally = self.sp.get_tally(name='tallies_on_mesh')
            flux_df = flux_tally.get_pandas_dataframe()

            # The dataframe only has the indices for the mesh geometry, not their actual values
            # We will replace the indices with values
            x = np.linspace(-plasma_outerRadius, plasma_outerRadius, self.N_points)
            y = np.linspace(-plasma_outerRadius, plasma_outerRadius, self.N_points)
            z = np.linspace(-plasma_length, plasma_length, self.N_points)

            fig, ax = plt.subplots()
            ax.set_aspect('equal', adjustable='box')
            if section == 'perp':
                slice_df = flux_df[flux_df[('mesh 1', 'z')] == (self.N_points - 1) / 2]
                [X, Y] = np.meshgrid(x, y)
                values = slice_df['mean'].to_numpy().reshape((self.N_points, self.N_points))
                plt.contourf(X, Y, values, levels=15)
                plt.xlabel('x (cm)')
            elif section == 'parallel':
                slice_df = flux_df[flux_df[('mesh 1', 'x')] == (self.N_points - 1) / 2]
                [Z, Y] = np.meshgrid(z, y)
                values = slice_df['mean'].to_numpy().reshape((self.N_points, self.N_points)).T
                plt.contourf(Z, Y, values, levels=15)
                plt.xlabel('z (cm)')
            else:
                print('Please type correct value for section, either perp or parallel')

            plt.colorbar(label=f'Flux')
            plt.ylabel('y (cm)')
            plt.savefig(f'{self.directory}/flux_{section}.png', dpi=200)
            plt.show()
        else:
            print('Did not create mesh tally, so not plotting flux')