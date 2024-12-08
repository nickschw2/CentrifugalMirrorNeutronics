from constants import *
import openmc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class MaterialDamageSource():
    def __init__(self, fuel='D-T', angle=0, material_name='BN', particles=10000, batches=100, plot_flux=False):
        self.fuel = fuel
        self.angle = angle
        self.material_name = material_name
        self.particles = particles
        self.batches = batches
        self.plot_flux = plot_flux

        self.set_materials()
        self.set_geometry()
        self.set_source()
        self.set_tallies()
        self.run_settings()

    def set_materials(self):
        if self.material_name == 'BN':
            mat = openmc.Material(name='BN')
            mat.add_element('B', 0.5)
            mat.add_element('N', 0.5)
            mat.set_density('g/cm3', 2.1)  # Density can vary based on form (h-BN is typically around 2.1 g/cm³)
            # Add thermal scattering data for boron nitride
            # mat.add_s_alpha_beta('c_BN')
        elif self.material_name == 'SiC':
            mat = openmc.Material(name='SiC')
            mat.add_element('Si', 0.5)
            mat.add_element('C', 0.5)
            mat.set_density('g/cm3', 3.2)
            # Add thermal scattering data for SiC
            # mat.add_s_alpha_beta('c_C_in_SiC')
            # mat.add_s_alpha_beta('c_Si_in_SiC')
        elif self.material_name == 'W':
            mat = openmc.Material(name='W')
            mat.add_element('W', 1.0)
            mat.set_density('g/cm3', 19.3)
        else:
            raise ValueError(f"Unknown material: {self.material}")
        self.target_material = mat
        self.materials = openmc.Materials([mat])

    def set_geometry(self):
        # Create the bounding surfaces for the target
        min_x = openmc.XPlane(-target_width/2, boundary_type='periodic')
        max_x = openmc.XPlane(target_width/2, boundary_type='periodic')
        min_y = openmc.YPlane(-target_height/2, boundary_type='periodic')
        max_y = openmc.YPlane(target_height/2, boundary_type='periodic')
        min_z = openmc.ZPlane(0, boundary_type='vacuum')
        max_z = openmc.ZPlane(target_thickness, boundary_type='vacuum')

        # Associate the periodic surfaces
        min_x.periodic_surface = max_x
        min_y.periodic_surface = max_y

        target_region = +min_x & -max_x & +min_y & -max_y & +min_z & -max_z
        target_cell = openmc.Cell(fill=self.target_material, region=target_region)

        # Define a larger void region around the target
        outer_box = openmc.model.RectangularParallelepiped(-target_width, target_width,
                                                           -target_height, target_height,
                                                           -target_thickness, 2*target_thickness, boundary_type='vacuum')
        void_region = ~target_region & -outer_box
        void_cell = openmc.Cell(region=void_region)
        
        universe = openmc.Universe(cells=[target_cell, void_cell])
        self.geometry = openmc.Geometry(universe)

    def set_source(self):
        fuel_energies = {'D-D': 2.45e6, 'D-T': 14.1e6} # eV
        if self.fuel not in fuel_energies:
            raise ValueError(f"Unknown fuel type: {self.fuel}")
        
        energy = openmc.stats.Discrete([fuel_energies[self.fuel]], [1.0])
        
        angle_rad = np.radians(self.angle)
        direction = openmc.stats.Monodirectional((np.sin(angle_rad), 0, np.cos(angle_rad)))
        space = openmc.stats.Point()
        
        # Create a planar source just in front of the target
        self.source = openmc.IndependentSource(space=space, angle=direction, energy=energy)

    def set_tallies(self):
        mesh = openmc.RegularMesh()
        mesh.dimension = [1, 1, N_mesh_points]
        mesh.lower_left = [-target_width/2, -target_height/2, 0]
        mesh.upper_right = [target_width/2, target_height/2, target_thickness]

        tally = openmc.Tally(name='damage')
        tally.filters = [openmc.MeshFilter(mesh)]
        tally.scores = ['damage-energy', 'absorption', 'flux']

        self.tallies = openmc.Tallies([tally])

        # Create 2D mesh to plot the flux map
        if self.plot_flux:
            flux_mesh = openmc.RegularMesh()
            flux_mesh.dimension = [N_mesh_points, 1, N_mesh_points]
            flux_mesh.lower_left = [-target_width/2, -target_height/2, 0]
            flux_mesh.upper_right = [target_width/2, target_height/2, target_thickness]
            flux_tally = openmc.Tally(name='flux_map')
            flux_tally.filters = [openmc.MeshFilter(flux_mesh)]
            flux_tally.scores = ['damage-energy', 'absorption', 'flux']
            self.tallies.append(flux_tally)


    def run_settings(self):
        self.settings = openmc.Settings()
        self.settings.batches = self.batches
        self.settings.particles = self.particles
        self.settings.run_mode = "fixed source"
        self.settings.source = self.source
        # Extend energy range to capture thermal effects
        self.settings.energy_mode = 'continuous-energy'
        self.settings.cutoff = {'energy_neutron': 1}  # Cut off energy

    def run(self, directory):
        self.directory = directory
        self.model = openmc.model.Model(self.geometry, self.materials, self.settings, self.tallies)
        try:
            self.sp_filename = self.model.run(cwd=self.directory, threads=n_threads)
        except RuntimeError:
            print('Too many particles lost')

    def analyze_results(self):
        # open the results file
        self.sp = openmc.StatePoint(self.sp_filename)

        # access the tally using pandas dataframes
        tally = self.sp.get_tally(name='damage')
        dpa = tally.get_slice(scores=['damage-energy']).mean.flatten()
        abs = tally.get_slice(scores=['absorption']).mean.flatten()
        flux = tally.get_slice(scores=['flux']).mean.flatten()
        dpa_std = tally.get_slice(scores=['damage-energy']).std_dev.flatten()
        abs_std = tally.get_slice(scores=['absorption']).std_dev.flatten()
        flux_std = tally.get_slice(scores=['flux']).std_dev.flatten()
        z = np.linspace(0, target_thickness, N_mesh_points)

        # Calculate penetration depth (e.g., depth at which flux reduces to 1% of surface flux)
        try:
            index = np.where(flux < 0.01 * flux[0])[0][0]
            depth = z[index]
        except:
            depth = 0

        # Calculate the total damage and absorption by integrating the signals
        dpa_tot = np.trapz(dpa, z)
        abs_tot = np.trapz(abs, z)

        columns = ['dpa', 'abs', 'flux', 'depth', 'dpa_tot', 'abs_tot', 'dpa_std', 'abs_std', 'flux_std']
        result = pd.DataFrame([[dpa, abs, flux, depth, dpa_tot, abs_tot, dpa_std, abs_std, flux_std]], columns=columns)

        return pd.DataFrame(result)
    
    def plot_flux_map(self):
        if self.plot_flux:
            # open the results file
            self.sp = openmc.StatePoint(self.sp_filename)
            flux_tally = self.sp.get_tally(name='flux_map')
            flux_df = flux_tally.get_pandas_dataframe()

            # The dataframe only has the indices for the mesh geometry, not their actual values
            # We will replace the indices with values
            mesh = flux_tally.filters[0].mesh
            nx, ny, nz = mesh.dimension
            x, dx = np.linspace(mesh.lower_left[0], mesh.upper_right[0], nx, retstep=True)
            z, dz = np.linspace(mesh.lower_left[2], mesh.upper_right[2], nz, retstep=True)
            dy = mesh.upper_right[1] - mesh.lower_left[1]

            # Normalize to particle / cm^2-s because flux comes in particle-cm
            # By setting the source strength, all results are totals, not per source particle
            mesh_cell_volume = dx * dy * dz
            # flux_df['mean'] = flux_df['mean'] / mesh_cell_volume
            dpa = flux_tally.get_slice(scores=['damage-energy']).mean.flatten()
            abs = flux_tally.get_slice(scores=['absorption']).mean.flatten()
            flux = flux_tally.get_slice(scores=['flux']).mean.flatten()

            plt.contourf(x, z, np.reshape(flux, (len(x), len(z))))
            plt.colorbar(label=r'Flux $\left( \frac{\#}{ \mathrm{cm}^2 \mathrm{s} } \right)$')
            plt.xlabel('x (cm)')
            plt.ylabel('z (cm)')
            plt.savefig(f'{root}/neutron_flux_materials.png', dpi=300)
            plt.show()