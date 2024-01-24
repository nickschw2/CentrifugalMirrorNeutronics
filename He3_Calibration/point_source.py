import openmc
import numpy as np
import matplotlib.pyplot as plt
import neutronics_material_maker as nmm
from constants import *
import pickle

class Point_Source():
    def __init__(self, activity=1, shielding=False, particles=250000, distance=20):
        self.activity = activity
        self.shielding = shielding
        self.particles = particles
        self.distance = distance

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
        if not self.shielding:
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

        self.universe = openmc.Universe(cells=[self.void_cell, self.He3_cell, self.detector_cell, self.HDPE_cell])
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

    def set_tallies(self):
        self.tallies = openmc.Tallies()

        cell_tally = openmc.Tally(name="tally_in_cell")
        cell_filter = openmc.CellFilter(self.He3_cell)
        cell_tally.scores = ["absorption"]
        cell_tally.filters = [cell_filter]
        self.tallies.append(cell_tally)

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


